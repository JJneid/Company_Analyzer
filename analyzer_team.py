import asyncio
from typing import List, Dict
import pandas as pd
from datetime import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
import json
import re
from urllib.parse import quote_plus
import time as time

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

def save_report(filename: str, data: Dict = None) -> str:
    try:
        # Ensure the results directory exists
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Add .txt extension if not present
        if not filename.endswith('.txt'):
            filename += '.txt'
            
        filepath = os.path.join(results_dir, filename)
        
        # Save both text report and JSON data
        with open(filepath, 'w') as f:
            f.write('Startup Analysis Report\n')
            f.write('=====================\n\n')
            
            if data:
                # Write formatted report
                for company, analysis in data.items():
                    f.write(f'Company: {company}\n')
                    f.write('=' * (len(company) + 9) + '\n\n')
                    
                    for section in ['technical', 'market', 'content']:
                        if section in analysis:
                            f.write(f'{section.title()} Analysis\n')
                            f.write('-' * (len(section) + 9) + '\n')
                            f.write(json.dumps(analysis[section], indent=2))
                            f.write('\n\n')
                            
        # Also save raw JSON data
        json_filepath = filepath.replace('.txt', '.json')
        with open(json_filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        return f'Reports saved successfully at {filepath} and {json_filepath}'
        
    except Exception as e:
        return f'Error saving report: {str(e)}'

# Create function tools
github_tool = FunctionTool(github_analysis, description="Analyze GitHub repository metrics")
market_tool = FunctionTool(startup_market_analysis, description="Analyze startup market presence and news")
content_tool = FunctionTool(content_analysis, description="Analyze website content for startup signals")
changelog_tool = FunctionTool(analyze_changelog, description="Analyze changelog or updates section for development activity")
report_tool = FunctionTool(
    save_report, 
    description="Generate and save analysis report to a file",
    name="save_report"  # Explicitly set the name
)

# Initialize model client
model_client = OpenAIChatCompletionClient(model="gpt-4o")

# Update the technical agent to properly structure its output
technical_agent = AssistantAgent(
    name="TechnicalAnalyst",
    model_client=model_client,
    tools=[github_tool, changelog_tool],
    system_message="""You are a technical analyst specializing in startup technology stacks and development patterns.
    For each analysis:
    1. Use github_tool to analyze repository if provided
    2. If GitHub isn't accessible, use changelog_tool
    3. Structure your findings as:
       {
           "technical_analysis": {
               "github_metrics": <github_tool_results>,
               "changelog_analysis": <changelog_tool_results>,
               "summary": "your summary here"
           }
       }
    4. Include 'TECHNICAL_COMPLETE' after your analysis.
    """
)

# Update market agent to structure its output
market_agent = AssistantAgent(
    name="StartupAnalyst",
    model_client=model_client,
    tools=[market_tool],
    system_message="""You are a startup analyst specializing in market research.
    For each analysis:
    1. Use market_tool to analyze market presence
    2. Structure your findings as:
       {
           "market_analysis": {
               "market_data": <market_tool_results>,
               "summary": "your summary here"
           }
       }
    3. Include 'MARKET_COMPLETE' after your analysis.
    """
)

# Update content agent to structure its output
content_agent = AssistantAgent(
    name="ContentAnalyst",
    model_client=model_client,
    tools=[content_tool],
    system_message="""You are a content analyst specializing in online presence.
    For each analysis:
    1. Use content_tool to analyze website content
    2. Structure your findings as:
       {
           "content_analysis": {
               "website_data": <content_tool_results>,
               "summary": "your summary here"
           }
       }
    3. Include 'CONTENT_COMPLETE' after your analysis.
    """
)

# Update report agent to properly handle the collected data
report_agent = AssistantAgent(
    name="ReportGenerator",
    model_client=model_client,
    tools=[report_tool],
    system_message="""You are a report generator specializing in startup analysis reports.
    For each report:
    1. Collect all previous analyses
    2. Use save_report tool to generate report with format:
       save_report(
           filename="company_name_report.txt",
           data={
               "company_name": {
                   "technical": technical_analysis,
                   "market": market_analysis,
                   "content": content_analysis
               }
           }
       )
    3. Include 'TERMINATE' only after successful save.
    """
)


# Define termination conditions for proper workflow control
termination_condition = TextMentionTermination("TERMINATE")
technical_complete = TextMentionTermination("TECHNICAL_COMPLETE")
market_complete = TextMentionTermination("MARKET_COMPLETE")
content_complete = TextMentionTermination("CONTENT_COMPLETE")

# Create the team
analysis_team = RoundRobinGroupChat(
    participants=[technical_agent, market_agent, content_agent, report_agent],
    termination_condition=termination_condition,
    max_turns=12
)



import asyncio
import pandas as pd
from typing import Dict, List

async def process_companies(csv_path: str) -> Dict:
    """
    Process companies from a CSV file and run analysis on each one.
    
    Args:
        csv_path: Path to CSV file with columns: company_name, website, github_url
    Returns:
        Dict containing analysis results for all companies
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    results = {}
    
    # Validate required columns
    required_columns = ['company_name', 'website', 'github_url']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Process each company
    for _, row in df.iterrows():
        company_name = row['company_name']
        print(f"\nAnalyzing {company_name}...")
        
        # Prepare the analysis task message
        task_message = f"""Analyze the following company:
        Company Name: {company_name}
        Website: {row['website']}
        GitHub URL: {row['github_url']}
        
        Please perform technical, market, and content analysis, then generate a report."""
        
        try:
            # Create a new team instance for each company
            analysis_team = RoundRobinGroupChat(
                participants=[technical_agent, market_agent, content_agent, report_agent],
                termination_condition=termination_condition,
                max_turns=12
            )
            
            # Run the analysis
            result = await analysis_team.run(task=task_message)
            results[company_name] = result
            
            print(f"✓ Completed analysis for {company_name}")
            
            # Add delay to respect rate limits
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"✗ Error analyzing {company_name}: {str(e)}")
            results[company_name] = {"error": str(e)}
    
    return results

# Main execution function
async def main():
    """Main execution function for the startup analysis system."""
    try:
        # Replace with your CSV path
        csv_path = "company_list.csv"
        
        print("Starting startup analysis system...")
        print(f"Reading companies from: {csv_path}")
        
        results = await process_companies(csv_path)
        
        print("\nAnalysis complete!")
        print(f"Processed {len(results)} companies")
        
        # Save final results
        save_report("final_analysis_report", results)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

# Run the analysis
if __name__ == "__main__":
    asyncio.run(main())