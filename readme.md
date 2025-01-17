# Startup Analysis System

A comprehensive tool for analyzing startups using multiple data sources, including GitHub metrics, market presence, and website content analysis.

## Overview

This system performs automated analysis of startups by examining:
- Technical metrics from GitHub repositories
- Market presence and news coverage
- Website content and signals
- Development activity through changelog analysis

## Quick Start

1. Set up your environment variables in a `.env` file:
```
GITHUB_TOKEN=your_github_token
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id
PRODUCTHUNT_TOKEN=your_producthunt_token
```

2. Create a CSV file named `company_list.csv` with the following columns:
```csv
company_name,website,github_url
ExampleCorp,https://example.com,https://github.com/example/repo
```

3. Install required dependencies:
```bash
# Core dependencies
pip install pandas beautifulsoup4 requests python-dotenv aiohttp asyncio

# Autogen dependencies
pip install -U "autogen-agentchat"
pip install "autogen-ext[openai]"

```

4. Run the analysis:
```bash
python startup_analyzer.py
```

## Output

The system generates two types of reports in the `results` directory:
- A human-readable `.txt` report
- A machine-readable `.json` file containing all raw data

## Features

- GitHub Repository Analysis
  - Stars, forks, and issues metrics
  - Development activity
  - Community health indicators

- Market Analysis
  - TechCrunch coverage
  - Google Search results
  - Product Hunt presence (if available)

- Website Content Analysis
  - Startup signal indicators
  - Team and hiring signals
  - Contact information
  - Social media presence

- Changelog Analysis
  - Update frequency
  - Recent changes
  - Development patterns

## Rate Limiting

The system includes built-in rate limiting to respect API constraints. Default rate is 1 request per second, which can be adjusted in the `RateLimiter` class.

## Error Handling

- Failed analyses for individual companies won't stop the entire process
- Errors are logged and included in the final report
- Each company gets a fresh analysis attempt

## Notes

- Ensure your API tokens have sufficient permissions
- The system requires internet access to fetch data
- Analysis time depends on the number of companies in the CSV
- Some features may be limited if certain API tokens are not provided