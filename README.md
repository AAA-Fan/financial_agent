# Multi-Agent Financial Advisory System

A sophisticated financial advisory system built with LangChain that uses multiple AI agents to analyze stocks and provide investment recommendations.

## System Architecture

The system consists of three specialized agents:

1. **Historical Data Analysis Agent**: Fetches and analyzes historical stock data (1 week) including price trends, volatility, and volume patterns.

2. **News Sentiment Analysis Agent**: Fetches the latest news from the internet and performs sentiment analysis to identify positive and negative factors affecting the stock.

3. **Supervisor Agent**: Coordinates both agents, synthesizes their analyses, and provides final investment recommendations (BUY/SELL/HOLD) with risk assessment.

## Features

- **Multi-Agent Architecture**: Specialized agents working in parallel
- **Historical Analysis**: Technical analysis of stock price movements and trends
- **News Sentiment**: Real-time news fetching and sentiment analysis
- **Intelligent Recommendations**: AI-powered synthesis and investment advice
- **Comprehensive Reports**: Detailed reports saved to files

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- (Optional) Tavily API key for enhanced news search

### Setup

1. Clone or navigate to the project directory:
```bash
cd financailagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

For enhanced news search, you can optionally add:
```
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

### Basic Usage

Run the main script:
```bash
python main.py
```

Enter a stock symbol when prompted (e.g., AAPL, GOOGL, MSFT, TSLA).

### Programmatic Usage

You can also use the agents programmatically:

```python
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from agents import HistoricalAnalysisAgent, NewsSentimentAgent, SupervisorAgent

load_dotenv()
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)

# Initialize agents
historical_agent = HistoricalAnalysisAgent(llm)
news_agent = NewsSentimentAgent(llm)
supervisor_agent = SupervisorAgent(llm)

# Analyze a stock
stock_symbol = "AAPL"

historical_result = historical_agent.analyze(stock_symbol)
news_result = news_agent.analyze(stock_symbol)
recommendation = supervisor_agent.make_recommendation(
    historical_result, 
    news_result, 
    stock_symbol
)

print(supervisor_agent.format_final_report(recommendation))
```

## Agent Details

### Historical Data Analysis Agent

- Fetches 7 days of historical stock data using yfinance
- Calculates key metrics:
  - Price changes and percentage changes
  - Volatility measures
  - Volume analysis
  - Trend identification
  - Support and resistance levels
- Identifies bullish or bearish patterns

### News Sentiment Analysis Agent

- Fetches latest news articles from multiple sources
- Uses DuckDuckGo and Tavily search APIs
- Performs detailed sentiment analysis:
  - Overall sentiment classification (Positive/Negative/Neutral)
  - Sentiment scoring
  - Key positive and negative factors
  - Impact assessment on stock price
  - Market and sector considerations

### Supervisor Agent

- Synthesizes information from both agents
- Identifies alignment or conflicts between technical and fundamental analysis
- Provides clear recommendations:
  - BUY/SELL/HOLD decision
  - Risk assessment
  - Supporting factors
  - Potential price targets
  - Important caveats

## Output

The system generates:
1. Console output with real-time progress
2. Detailed final report displayed in terminal
3. Saved report file: `report_{SYMBOL}_{TIMESTAMP}.txt`

Report includes:
- Executive summary and recommendation
- Detailed historical analysis
- Comprehensive news sentiment analysis
- Risk assessment and key considerations

## Dependencies

- **langchain**: Core framework for agent orchestration
- **langchain-openai**: OpenAI LLM integration
- **langchain-community**: Community tools and integrations
- **langgraph**: Advanced multi-agent workflows
- **yfinance**: Stock market data
- **duckduckgo-search**: News search
- **tavily-python**: Enhanced news search (optional)
- **pandas, numpy**: Data analysis
- **beautifulsoup4, feedparser, newspaper3k**: Web scraping

## Configuration

You can customize the system by:
- Adjusting LLM temperature in `main.py` (default: 0.3 for consistency)
- Changing the number of days for historical analysis (default: 7)
- Modifying the number of news articles fetched (default: 10)
- Using different LLM models

## Limitations

- Requires internet connection for news fetching
- API rate limits may apply for OpenAI and search APIs
- Historical data depends on market hours and data availability
- News sentiment is based on available sources and may not capture all relevant news
- Recommendations are for informational purposes only and should not be considered as financial advice

## Troubleshooting

1. **API Key Errors**: Ensure your `.env` file contains a valid OpenAI API key
2. **No News Found**: Try adding a Tavily API key for better news search results
3. **Data Fetching Errors**: Check your internet connection and verify the stock symbol is correct
4. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This system provides AI-generated financial analysis and recommendations for informational purposes only. It should not be considered as professional financial advice. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results, and investments carry inherent risks.

