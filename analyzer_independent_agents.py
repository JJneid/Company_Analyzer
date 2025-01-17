import asyncio
from typing import List, Dict
import pandas as pd
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
import json
import re
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_second=1):
        self.calls_per_second = calls_per_second
        self.last_call = 0
    
    async def wait(self):
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_call
        if time_since_last < (1 / self.calls_per_second):
            await asyncio.sleep((1 / self.calls_per_second) - time_since_last)
        self.last_call = current_time

rate_limiter = RateLimiter()

# Tool Implementations
def github_analysis(repo_url: str) -> Dict:
    """
    Analyze GitHub repository metrics with enhanced startup-focused metrics.
    Returns None if repository is private or not accessible.
    """
    if not repo_url:  # Handle cases where no GitHub URL is provided
        return None
    try:
        # Extract owner and repo name from URL
        parts = repo_url.split('/')
        owner = parts[-2]
        repo = parts[-1]
        
        headers = {'Authorization': f'token {os.getenv("GITHUB_TOKEN")}'}
        
        # Get repository data
        api_url = f'https://api.github.com/repos/{owner}/{repo}'
        response = requests.get(api_url, headers=headers)
        data = response.json()
        
        # Get commit frequency
        commits_url = f'https://api.github.com/repos/{owner}/{repo}/stats/participation'
        commits_response = requests.get(commits_url, headers=headers)
        commit_data = commits_response.json()
        
        # Get contributor statistics
        contributors_url = f'https://api.github.com/repos/{owner}/{repo}/contributors'
        contributors_response = requests.get(contributors_url, headers=headers)
        contributor_data = contributors_response.json()
        
        return {
            'repository_metrics': {
                'stars': data.get('stargazers_count'),
                'forks': data.get('forks_count'),
                'open_issues': data.get('open_issues_count'),
                'watchers': data.get('subscribers_count'),
                'last_update': data.get('updated_at'),
                'created_at': data.get('created_at')
            },
            'development_activity': {
                'recent_commits': sum(commit_data.get('owner', [])) if isinstance(commit_data, dict) else None,
                'total_contributors': len(contributor_data) if isinstance(contributor_data, list) else None,
                'primary_language': data.get('language'),
                'all_languages': data.get('languages_url')
            },
            'community_health': {
                'has_wiki': data.get('has_wiki'),
                'has_pages': data.get('has_pages'),
                'has_projects': data.get('has_projects'),
                'description': data.get('description')
            }
        }
    except Exception as e:
        return {'error': str(e)}

async def startup_market_analysis(company_name: str) -> Dict:
    """
    Analyze startup market data using multiple sources including TechCrunch, 
    Crunchbase mentions, and other startup-focused sources.
    """
    try:
        await rate_limiter.wait()
        results = {'news_analysis': [], 'funding_data': {}, 'market_presence': {}}
        
        # TechCrunch search
        techcrunch_url = f'https://techcrunch.com/search/{quote_plus(company_name)}'
        response = requests.get(techcrunch_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('a', {'class': 'post-block__title__link'})
        
        results['news_analysis'].extend([
            {
                'source': 'TechCrunch',
                'title': article.text.strip(),
                'url': article['href'],
                'date': article.find_parent().find('time').text if article.find_parent().find('time') else None
            } for article in articles[:5]  # Latest 5 articles
        ])
        
        # Use Google Search API for broader coverage
        if os.getenv('GOOGLE_API_KEY') and os.getenv('GOOGLE_SEARCH_ENGINE_ID'):
            search_query = f'{company_name} startup funding OR investment OR acquisition'
            google_url = 'https://customsearch.googleapis.com/customsearch/v1'
            params = {
                'key': os.getenv('GOOGLE_API_KEY'),
                'cx': os.getenv('GOOGLE_SEARCH_ENGINE_ID'),
                'q': search_query
            }
            response = requests.get(google_url, params=params)
            search_results = response.json().get('items', [])
            
            results['market_presence']['recent_mentions'] = [
                {
                    'title': item.get('title'),
                    'link': item.get('link'),
                    'snippet': item.get('snippet')
                } for item in search_results[:5]
            ]
        
        # Product Hunt data (if available)
        producthunt_url = f'https://api.producthunt.com/v2/api/graphql'
        if os.getenv('PRODUCTHUNT_TOKEN'):
            headers = {'Authorization': f'Bearer {os.getenv("PRODUCTHUNT_TOKEN")}'}
            query = '''
            query($company: String!) {
                posts(search: $company, first: 5) {
                    edges {
                        node {
                            name
                            tagline
                            votesCount
                            website
                            createdAt
                        }
                    }
                }
            }
            '''
            response = requests.post(
                producthunt_url,
                headers=headers,
                json={'query': query, 'variables': {'company': company_name}}
            )
            if response.status_code == 200:
                results['market_presence']['product_hunt'] = response.json()
        
        return results
    except Exception as e:
        return {'error': str(e)}

def analyze_changelog(url: str) -> Dict:
    """
    Analyze changelog or updates section of a website for development activity signals.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Common changelog URL patterns
        changelog_patterns = [
            '/changelog', '/updates', '/releases', '/whats-new', 
            '/news', '/blog/changelog', '/release-notes'
        ]
        
        # Look for changelog links
        changelog_links = []
        for link in soup.find_all('a', href=True):
            if any(pattern in link['href'].lower() for pattern in changelog_patterns):
                changelog_links.append(link['href'])
        
        # If found changelog links, analyze the most recent one
        changelog_content = {}
        if changelog_links:
            changelog_url = changelog_links[0]
            if not changelog_url.startswith('http'):
                changelog_url = url.rstrip('/') + '/' + changelog_url.lstrip('/')
            
            changelog_response = requests.get(changelog_url)
            changelog_soup = BeautifulSoup(changelog_response.content, 'html.parser')
            
            # Extract dates and updates
            dates = []
            updates = []
            
            # Look for common changelog patterns
            for element in changelog_soup.find_all(['h1', 'h2', 'h3', 'time']):
                if re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', element.text):
                    dates.append(element.text.strip())
                    update_content = []
                    next_elem = element.find_next_sibling()
                    while next_elem and next_elem.name not in ['h1', 'h2', 'h3', 'time']:
                        if next_elem.text.strip():
                            update_content.append(next_elem.text.strip())
                        next_elem = next_elem.find_next_sibling()
                    updates.append(' '.join(update_content))
            
            changelog_content = {
                'found': True,
                'url': changelog_url,
                'entries': [{'date': d, 'content': u} for d, u in zip(dates[:5], updates[:5])],
                'update_frequency': len(dates),
                'last_update': dates[0] if dates else None
            }
        else:
            changelog_content = {
                'found': False,
                'message': 'No changelog section found'
            }
        
        return changelog_content
        
    except Exception as e:
        return {'error': str(e)}

def content_analysis(url: str) -> Dict:
    """
    Enhanced web scraping and content analysis focused on startup signals.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract all text content
        text_content = soup.get_text()
        
        # Look for specific startup indicators
        indicators = {
            'hiring': len(re.findall(r'hir(e|ing)|career|job', text_content, re.I)),
            'product': len(re.findall(r'product|feature|solution', text_content, re.I)),
            'technology': len(re.findall(r'tech|platform|API|stack', text_content, re.I)),
            'customers': len(re.findall(r'customer|client|user', text_content, re.I))
        }
        
        # Extract contact information
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = list(set(re.findall(email_pattern, text_content)))
        
        return {
            'basic_info': {
                'title': soup.title.string if soup.title else None,
                'meta_description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else None,
            },
            'content_analysis': {
                'word_count': len(text_content.split()),
                'indicator_mentions': indicators,
            },
            'startup_signals': {
                'team_page_exists': bool(soup.find_all(string=re.compile(r'team|about us', re.I))),
                'blog_exists': bool(soup.find_all(string=re.compile(r'blog|news|updates', re.I))),
                'contact_info': {
                    'emails': emails,
                    'social_links': [link['href'] for link in soup.find_all('a', href=re.compile(r'linkedin\.com|twitter\.com|github\.com'))]
                }
            }
        }
    except Exception as e:
        return {'error': str(e)}

def save_report(data: Dict, filename: str) -> str:
    """
    Enhanced report generation with startup-focused insights.
    """
    try:
        os.makedirs('results', exist_ok=True)
        filepath = os.path.join('results', filename)
        
        with open(filepath, 'w') as f:
            f.write('Startup Competitive Analysis Report\n')
            f.write('================================\n\n')
            
            for company, analysis in data.items():
                f.write(f'Company: {company}\n')
                f.write('=' * (len(company) + 9) + '\n\n')
                
                # Technical Analysis
                if 'technical' in analysis:
                    f.write('Technical Profile\n')
                    f.write('-----------------\n')
                    tech_data = analysis['technical']
                    if 'repository_metrics' in tech_data:
                        f.write('GitHub Activity:\n')
                        for metric, value in tech_data['repository_metrics'].items():
                            f.write(f'  - {metric}: {value}\n')
                    f.write('\n')
                
                # Market Analysis
                if 'market' in analysis:
                    f.write('Market Presence\n')
                    f.write('---------------\n')
                    market_data = analysis['market']
                    if 'news_analysis' in market_data:
                        f.write('Recent News Coverage:\n')
                        for article in market_data['news_analysis']:
                            f.write(f'  - {article["title"]}\n')
                            f.write(f'    Source: {article["source"]}, Date: {article["date"]}\n')
                    f.write('\n')
                
                # Web Analysis
                if 'web' in analysis:
                    f.write('Online Presence\n')
                    f.write('---------------\n')
                    web_data = analysis['web']
                    if 'startup_signals' in web_data:
                        f.write('Key Indicators:\n')
                        for category, signals in web_data['startup_signals'].items():
                            f.write(f'  - {category}: {signals}\n')
                    f.write('\n')
                
                f.write('\n' + '=' * 80 + '\n\n')
                
        return f'Report saved successfully at {filepath}'
    except Exception as e:
        return f'Error saving report: {str(e)}'

# Create function tools
github_tool = FunctionTool(github_analysis, description="Analyze GitHub repository metrics")
market_tool = FunctionTool(startup_market_analysis, description="Analyze startup market presence and news")
content_tool = FunctionTool(content_analysis, description="Analyze website content for startup signals")
changelog_tool = FunctionTool(analyze_changelog, description="Analyze changelog or updates section for development activity")
report_tool = FunctionTool(save_report, description="Generate comprehensive startup analysis report")

# Initialize model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Define specialized agents
technical_agent = AssistantAgent(
    name="TechnicalAnalyst",
    model_client=model_client,
    tools=[github_tool, changelog_tool],
    system_message="""You are a technical analyst specializing in startup technology stacks and development patterns.
    Primary task is to analyze GitHub repositories when available. When GitHub data is not accessible:
    1. Look for and analyze changelog/updates sections
    2. Extract development velocity and feature rollout patterns from changelogs
    3. Identify technical stack information from release notes
    4. Assess product maturity through update frequency and complexity"""
)

market_agent = AssistantAgent(
    name="StartupAnalyst",
    model_client=model_client,
    tools=[market_tool],
    system_message="""You are a startup analyst specializing in market research and competitive intelligence.
    Analyze news coverage, funding events, and market presence to evaluate startup trajectory and potential."""
)

content_agent = AssistantAgent(
    name="ContentAnalyst",
    model_client=model_client,
    tools=[content_tool],
    system_message="""You are a content analyst specializing in startup messaging and online presence.
    Analyze website content, marketing messages, and growth signals."""
)

report_agent = AssistantAgent(
    name="ReportGenerator",
    model_client=model_client,
    tools=[report_tool],
    system_message="""You are a report generator specializing in startup analysis reports.
    Compile insights about technical capability, market presence, and growth indicators into actionable reports."""
)

# Create the team
analysis_team = RoundRobinGroupChat(
    participants=[technical_agent, market_agent, content_agent, report_agent],
    max_turns=12
)

async def run_analysis(companies_file: str):
    """
    Main function to run the startup competitive analysis with smart fallback to changelog analysis.
    """
    try:
        # Read companies from CSV
        df = pd.read_csv(companies_file)
        results = {}
        print("Starting Analysis")
        for _, row in df.iterrows():
            print('for company:', row['company_name'])
            # Initialize technical analysis with GitHub data if available
            technical_result = await technical_agent.run(
                task=f"""Analyze technical presence for {row['company_name']}:
                1. Try GitHub repository: {row.get('github_url', '')}
                2. If GitHub not available, analyze changelog at: {row['website']}
                3. Provide development activity assessment based on available data"""
            )
            print('technical result:', technical_result)
            company_data = {
                'technical': technical_result,
                'market': await market_agent.run(
                    task=f"Analyze market presence for startup: {row['company_name']}"
                ),
                'web': await content_agent.run(
                    task=f"Analyze startup website: {row['website']}"
                )
            }
            results[row['company_name']] = company_data
            print('company_data:', company_data)
        # Generate final report
        await report_agent.run(
            task=f"Generate comprehensive startup analysis report: {results}"
        )
        print("Analysis Completed")
    except Exception as e:
        print(f"Error running analysis: {str(e)}")

# Example usage
if __name__ == "__main__":
    asyncio.run(run_analysis("company_list.csv"))