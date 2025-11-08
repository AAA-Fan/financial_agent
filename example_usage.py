"""
Example usage of the Multi-Agent Financial Advisory System
Demonstrates programmatic usage of the agents
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from agents import HistoricalAnalysisAgent, IndicatorAnalysisAgent, NewsSentimentAgent, SupervisorAgent

import os
# proxy = 'http://127.0.0.1:7890'
# os.environ['http_proxy'] = proxy
# os.environ['https_proxy'] = proxy

def example_single_stock():
    """Example: Analyze a single stock"""
    load_dotenv()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.3
    )
    
    # Initialize agents
    historical_agent = HistoricalAnalysisAgent(llm)
    indicator_agent = IndicatorAnalysisAgent(llm)
    news_agent = NewsSentimentAgent(llm)
    supervisor_agent = SupervisorAgent(llm)
    
    # Analyze a stock
    stock_symbol = "AAPL"
    print(f"Analyzing {stock_symbol}...\n")
    
    # Get historical analysis
    print("Fetching historical data...")
    historical_result = historical_agent.analyze(stock_symbol)
    print(historical_result)

    # Get indicator analysis
    print("Fetching indicator data...")
    indicator_result = indicator_agent.analyze(stock_symbol)
    print(indicator_result)
    
    # Get news sentiment
    print("Fetching news and analyzing sentiment...")
    news_result = news_agent.analyze(stock_symbol)
    
    # Get recommendation
    print("Generating recommendation...")
    recommendation = supervisor_agent.make_recommendation(
        historical_result,
        indicator_result,
        news_result,
        stock_symbol
    )
    
    # Display report
    print(supervisor_agent.format_final_report(recommendation))


def example_multiple_stocks():
    """Example: Analyze multiple stocks"""
    load_dotenv()
    
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
    
    historical_agent = HistoricalAnalysisAgent(llm)
    news_agent = NewsSentimentAgent(llm)
    indicator_agent = IndicatorAnalysisAgent(llm)
    supervisor_agent = SupervisorAgent(llm)
    
    stocks = ["AAPL", "GOOGL", "MSFT"]
    
    for stock in stocks:
        print(f"\n{'='*80}")
        print(f"Analyzing {stock}")
        print('='*80)
        
        historical_result = historical_agent.analyze(stock)
        news_result = news_agent.analyze(stock)
        recommendation = supervisor_agent.make_recommendation(
            historical_result,
            indicator_agent,
            news_result,
            stock
        )
        
        print(supervisor_agent.format_final_report(recommendation))
        print("\n")


if __name__ == "__main__":
    # Run single stock example
    example_single_stock()
    
    # Uncomment to run multiple stocks example
    # example_multiple_stocks()

