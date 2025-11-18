"""
Supervisor Agent
Coordinates the other agents and makes final recommendations based on their analysis
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, List
import json


class SupervisorAgent:
    """Supervisor agent that coordinates analysis and provides recommendations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_prompt()
    
    def _setup_prompt(self):
        """Setup the prompt template for the supervisor"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior financial advisor and supervisor coordinating multiple analysis agents.
Your role is to:
1. Review analysis from historical data agent (price trends, volatility, technical indicators)
2. Review analysis from indicator data agent (RSI, other technical indicators)
3. Review analysis from news sentiment agent (positive/negative news, market sentiment)
4. Synthesize both analyses to form a comprehensive view
5. Provide clear, actionable investment recommendations

Consider:
- Alignment or conflicts between historical trends, technical indicators and news sentiment
- Risk factors and opportunities
- Short-term vs long-term implications
- Overall market conditions

Your recommendations should be clear and include:
- BUY/SELL/HOLD recommendation with reasoning
- Risk assessment
- Key factors supporting the recommendation
- Potential price targets or timeframes (if applicable)
- Important caveats or risks

Be objective and balanced in your assessment."""),
            ("human", "{input}")
        ])
    
    def make_recommendation(self, historical_analysis: Dict[str, Any], 
                           indicator_analysis: Dict[str, Any],
                           news_analysis: Dict[str, Any], 
                           stock_symbol: str) -> Dict[str, Any]:
        """
        Synthesize analyses and make final recommendation
        
        Args:
            historical_analysis: Results from historical data agent
            news_analysis: Results from news sentiment agent
            stock_symbol: Stock ticker symbol being analyzed
            
        Returns:
            Dictionary with final recommendation
        """
        # Prepare the input for the supervisor
        input_text = f"""
STOCK SYMBOL: {stock_symbol}

HISTORICAL DATA ANALYSIS:
{historical_analysis.get('analysis', 'No analysis available')}

INDICATOR DATA ANALYSIS:
{indicator_analysis.get('indicator_analysis', 'No indicator analysis available')}

NEWS SENTIMENT ANALYSIS:
{news_analysis.get('analysis', 'No analysis available')}

Based on the above analyses from both the historical data agent and news sentiment agent, 
please provide:
1. A comprehensive synthesis of both analyses
2. Your investment recommendation (BUY/SELL/HOLD) with clear reasoning
3. Risk assessment
4. Key supporting factors
5. Important considerations or warnings
"""
        
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"input": input_text})
            
            recommendation_text = result.content if hasattr(result, 'content') else str(result)
            
            return {
                "agent": "supervisor",
                "stock_symbol": stock_symbol,
                "recommendation": recommendation_text,
                "historical_status": historical_analysis.get("status", "unknown"),
                "indicator_status": indicator_analysis.get("status", "unknown"),
                "news_status": news_analysis.get("status", "unknown"),
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": "supervisor",
                "stock_symbol": stock_symbol,
                "recommendation": f"Error generating recommendation: {str(e)}",
                "historical_status": historical_analysis.get("status", "unknown"),
                "indicator_status": indicator_analysis.get("status", "unknown"),
                "news_status": news_analysis.get("status", "unknown"),
                "status": "error"
            }
    
    def format_final_report(self, recommendation: Dict[str, Any]) -> str:
        """
        Format the final recommendation into a readable report
        
        Args:
            recommendation: The recommendation dictionary from make_recommendation
            
        Returns:
            Formatted string report
        """
        stock_symbol = recommendation.get("stock_symbol", "UNKNOWN")
        rec_text = recommendation.get("recommendation", "No recommendation available")
        
        report = f"""
{'='*80}
FINANCIAL ADVISORY REPORT
{'='*80}
Stock Symbol: {stock_symbol}
Generated: {recommendation.get('timestamp', 'N/A')}

RECOMMENDATION:
{'-'*80}
{rec_text}

{'='*80}
Status: {recommendation.get('status', 'unknown')}
Historical Analysis: {recommendation.get('historical_status', 'unknown')}
Indicator Analysis: {recommendation.get('indicator_status', 'unknown')}
News Sentiment: {recommendation.get('news_status', 'unknown')}
{'='*80}
"""
        return report

