"""
Multi-Agent Financial Advisory System
Main orchestration script that coordinates all agents
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.historical_agent import HistoricalAnalysisAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.supervisor_agent import SupervisorAgent


def main():
    """Main function to run the multi-agent financial advisory system"""
    
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.3,  # Lower temperature for more consistent analysis
        api_key=api_key
    )
    
    # Initialize agents (set verbose=True to see detailed agent execution)
    print("Initializing agents...")
    verbose_mode = os.getenv("VERBOSE", "false").lower() == "true"
    historical_agent = HistoricalAnalysisAgent(llm, verbose=verbose_mode)
    news_agent = NewsSentimentAgent(llm, verbose=verbose_mode)
    supervisor_agent = SupervisorAgent(llm)
    
    print("Multi-Agent Financial Advisory System")
    print("=" * 60)
    
    # Get stock symbol from user
    stock_symbol = input("\nEnter stock symbol to analyze (e.g., AAPL, GOOGL, MSFT): ").strip().upper()
    
    if not stock_symbol:
        print("No stock symbol provided. Exiting.")
        return
    
    print(f"\nAnalyzing {stock_symbol}...")
    print("-" * 60)
    
    # Step 1: Historical Analysis
    print("\n[1/3] Historical Data Agent: Analyzing past week's performance...")
    historical_result = historical_agent.analyze(stock_symbol)
    if historical_result["status"] == "success":
        print("✓ Historical analysis completed")
    else:
        print("✗ Historical analysis encountered errors")
    
    # Step 2: News Sentiment Analysis
    print("\n[2/3] News Sentiment Agent: Fetching and analyzing latest news...")
    news_result = news_agent.analyze(stock_symbol)
    if news_result["status"] == "success":
        print("✓ News sentiment analysis completed")
    else:
        print("✗ News sentiment analysis encountered errors")
    
    # Step 3: Supervisor Recommendation
    print("\n[3/3] Supervisor Agent: Synthesizing analyses and generating recommendation...")
    recommendation = supervisor_agent.make_recommendation(
        historical_result,
        news_result,
        stock_symbol
    )
    recommendation["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display final report
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    final_report = supervisor_agent.format_final_report(recommendation)
    print(final_report)
    
    # Save report to file
    output_file = f"report_{stock_symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(output_file, 'w') as f:
        f.write(final_report)
        f.write("\n\nDETAILED ANALYSES:\n")
        f.write("=" * 80 + "\n\n")
        f.write("HISTORICAL ANALYSIS:\n")
        f.write(historical_result.get("analysis", "") + "\n\n")
        f.write("NEWS SENTIMENT ANALYSIS:\n")
        f.write(news_result.get("analysis", "") + "\n")
    
    print(f"\nFull report saved to: {output_file}")
    
    return recommendation


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

