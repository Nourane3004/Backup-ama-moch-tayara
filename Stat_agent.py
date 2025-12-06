# trading_quant_agent.py - Quantitative Trading Agent Compatible

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import json

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import concurrent.futures

warnings.filterwarnings('ignore')

class TradingQuantAgent:
    """
    Pure quantitative trading agent for orchestrator
    Technical and statistical analysis without personal portfolio management
    """

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._data_cache = {}
        self._analysis_cache = {}
        self._cache_duration = 300  # 5 minutes

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for the orchestrator
        """
        try:
            # Extract data from context
            market_data = context.get("market_data", {})
            query = context.get("query", "")

            # Detect symbols to analyze
            symbols = self._extract_symbols_from_context(market_data, query)

            if not symbols:
                symbols = ["SPY"]  # Default value

            # Analyze each symbol
            results = []
            for symbol in symbols[:5]:  # Limit to 5 symbols max
                try:
                    analysis = await self.analyze_symbol(symbol, context)
                    if analysis:
                        results.append(analysis)
                except Exception as e:
                    print(f"⚠️ Error analyzing {symbol}: {e}")
                    continue

            # Generate global signal
            global_signal = self._aggregate_signals(results)

            # Prepare response for orchestrator
            return {
                "agent": "QUANT-MODELER",
                "signal": global_signal.get("decision", "NEUTRAL").lower(),
                "score": global_signal.get("confidence", 0.5),
                "prediction_raw": global_signal.get("score", 0.0),
                "symbols_analyzed": [r["symbol"] for r in results],
                "detailed_analysis": results[:3],  # Keep first 3 details
                "reasoning": global_signal.get("reasons", []),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"❌ Error in TradingQuantAgent.run: {e}")
            # Return neutral signal by default in case of error
            return {
                "agent": "QUANT-MODELER",
                "signal": "neutral",
                "score": 0.5,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _extract_symbols_from_context(self, market_data: Dict, query: str) -> List[str]:
        """Extract symbols from context"""
        symbols = []

        # From market_data
        if market_data:
            symbols.extend(list(market_data.keys()))

        # From query (search for common symbols)
        symbol_patterns = {
            r'\b(apple|aapl)\b': 'AAPL',
            r'\b(tesla|tsla)\b': 'TSLA',
            r'\b(google|googl)\b': 'GOOGL',
            r'\b(amazon|amzn)\b': 'AMZN',
            r'\b(microsoft|msft)\b': 'MSFT',
            r'\b(meta|facebook|fb)\b': 'META',
            r'\b(nvidia|nvda)\b': 'NVDA',
            r'\b(netflix|nflx)\b': 'NFLX',
            r'\b(spy|sp500|s&p)\b': 'SPY',
            r'\b(qqq|nasdaq)\b': 'QQQ',
            r'\b(dia|dow)\b': 'DIA',
            r'\b(bitcoin|btc)\b': 'BTC-USD',
            r'\b(ethereum|eth)\b': 'ETH-USD'
        }

        if query:
            query_lower = query.lower()
            for pattern, symbol in symbol_patterns.items():
                import re
                if re.search(pattern, query_lower):
                    symbols.append(symbol)

        # Remove duplicates
        return list(set(symbols))

    async def analyze_symbol(self, symbol: str, context: Dict) -> Optional[Dict]:
        """Complete analysis of a symbol"""
        try:
            # Check cache
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
            if self.use_cache and cache_key in self._analysis_cache:
                cache_time, cached_data = self._analysis_cache[cache_key]
                if (datetime.now() - cache_time).seconds < self._cache_duration:
                    return cached_data

            # Fetch data
            data = await self._fetch_symbol_data(symbol)
            if data.empty:
                return None

            # Analyze with ARIMA-GARCH
            arima_result = await self._run_arima_analysis(data)

            # Technical analysis
            technical_result = self._technical_analysis(data)

            # Volatility analysis
            volatility_result = self._volatility_analysis(data)

            # Combine results
            signal = self._combine_signals(arima_result, technical_result, volatility_result)

            # Calculate confidence
            confidence = self._calculate_confidence(arima_result, technical_result, volatility_result)

            # Prepare result
            result = {
                "symbol": symbol,
                "current_price": float(data['Close'].iloc[-1]),
                "signal": signal,
                "confidence": confidence,
                "arima_forecast": arima_result.get("forecast", 0.0),
                "arima_std": arima_result.get("std", 0.0),
                "rsi": technical_result.get("rsi", 50),
                "momentum": technical_result.get("momentum", 0.0),
                "volatility": volatility_result.get("current_vol", 0.0),
                "is_high_vol": volatility_result.get("is_high_vol", False),
                "reasons": self._generate_reasons(signal, arima_result, technical_result, volatility_result),
                "timestamp": datetime.now().isoformat()
            }

            # Cache
            if self.use_cache:
                self._analysis_cache[cache_key] = (datetime.now(), result)

            return result

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

    async def _fetch_symbol_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Fetch symbol data"""
        try:
            # Check cache
            cache_key = f"data_{symbol}_{period}"
            if self.use_cache and cache_key in self._data_cache:
                cache_time, cached_data = self._data_cache[cache_key]
                if (datetime.now() - cache_time).seconds < self._cache_duration:
                    return cached_data

            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval="1d", prepost=False)

            if not hist.empty:
                # Cache
                if self.use_cache:
                    self._data_cache[cache_key] = (datetime.now(), hist)

            return hist

        except Exception as e:
            print(f"⚠️ Error fetching data {symbol}: {e}")
            return pd.DataFrame()

    async def _run_arima_analysis(self, data: pd.DataFrame) -> Dict:
        """Run ARIMA-GARCH analysis"""
        try:
            if len(data) < 20:
                return {"forecast": 0.0, "std": 0.0, "signal": 0}

            # Calculate log returns
            prices = data['Close'].dropna()
            if len(prices) < 2:
                return {"forecast": 0.0, "std": 0.0, "signal": 0}

            returns = np.log(prices / prices.shift(1)).dropna()

            # ARIMA
            arima_result = {"forecast": 0.0, "std": 0.0, "signal": 0}
            try:
                model = ARIMA(returns, order=(1, 0, 1))
                result = model.fit(method='css', disp=False)
                forecast = result.forecast(steps=3)[0]
                arima_result["forecast"] = float(forecast.mean())
                arima_result["std"] = float(forecast.std())
                arima_result["signal"] = 1 if forecast.mean() > 0 else -1 if forecast.mean() < 0 else 0
            except Exception as e:
                print(f"⚠️ ARIMA failed: {e}")

            return arima_result

        except Exception as e:
            print(f"⚠️ ARIMA-GARCH analysis failed: {e}")
            return {"forecast": 0.0, "std": 0.0, "signal": 0}

    def _technical_analysis(self, data: pd.DataFrame) -> Dict:
        """Technical analysis"""
        try:
            if len(data) < 15:
                return {"rsi": 50, "momentum": 0.0, "sma_ratio": 1.0}

            prices = data['Close']

            # RSI
            rsi = self._calculate_rsi(prices)

            # Momentum (10-day change)
            momentum = 0.0
            if len(prices) >= 11:
                momentum = ((prices.iloc[-1] / prices.iloc[-11]) - 1) * 100

            # SMA Ratio
            sma_ratio = 1.0
            if len(prices) >= 20:
                sma_20 = prices.rolling(20).mean().iloc[-1]
                sma_ratio = prices.iloc[-1] / sma_20 if sma_20 > 0 else 1.0

            return {
                "rsi": rsi,
                "momentum": momentum,
                "sma_ratio": sma_ratio
            }

        except Exception as e:
            print(f"⚠️ Technical analysis failed: {e}")
            return {"rsi": 50, "momentum": 0.0, "sma_ratio": 1.0}

    def _volatility_analysis(self, data: pd.DataFrame) -> Dict:
        """Volatility analysis"""
        try:
            if len(data) < 10:
                return {"current_vol": 0.0, "is_high_vol": False}

            prices = data['Close']

            # Historical volatility (standard deviation of daily returns)
            if len(prices) >= 2:
                returns = np.log(prices / prices.shift(1)).dropna()
                current_vol = returns.std() * np.sqrt(252) * 100  # Annualized vol in %

                # Determine if high volatility (above 20% annualized)
                is_high_vol = current_vol > 20.0

                return {
                    "current_vol": float(current_vol),
                    "is_high_vol": is_high_vol
                }
            else:
                return {"current_vol": 0.0, "is_high_vol": False}

        except Exception as e:
            print(f"⚠️ Volatility analysis failed: {e}")
            return {"current_vol": 0.0, "is_high_vol": False}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0

            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            # Avoid division by zero
            loss = loss.replace(0, np.nan)
            rs = gain / loss

            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        except:
            return 50.0

    def _combine_signals(self, arima_result: Dict, technical_result: Dict, volatility_result: Dict) -> str:
        """Combine signals for final decision"""

        # Initialize scores
        buy_score = 0
        sell_score = 0

        # ARIMA signal
        arima_signal = arima_result.get("signal", 0)
        if arima_signal > 0:
            buy_score += 1
        elif arima_signal < 0:
            sell_score += 1

        # RSI
        rsi = technical_result.get("rsi", 50)
        if rsi < 30:
            buy_score += 1
        elif rsi > 70:
            sell_score += 1

        # Momentum
        momentum = technical_result.get("momentum", 0.0)
        if momentum > 2.0:
            buy_score += 0.5
        elif momentum < -2.0:
            sell_score += 0.5

        # SMA Ratio
        sma_ratio = technical_result.get("sma_ratio", 1.0)
        if sma_ratio > 1.02:
            buy_score += 0.5
        elif sma_ratio < 0.98:
            sell_score += 0.5

        # Volatility (penalty for high volatility)
        if volatility_result.get("is_high_vol", False):
            buy_score *= 0.7
            sell_score *= 0.7

        # Final decision
        if buy_score > sell_score + 0.5:
            return "BUY"
        elif sell_score > buy_score + 0.5:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_confidence(self, arima_result: Dict, technical_result: Dict, volatility_result: Dict) -> float:
        """Calculate signal confidence (0-1)"""

        confidence_factors = []

        # ARIMA confidence (based on signal strength)
        arima_forecast = abs(arima_result.get("forecast", 0.0))
        arima_conf = min(arima_forecast * 10, 1.0) if arima_forecast > 0 else 0.3
        confidence_factors.append(arima_conf)

        # RSI confidence (closer to extremes, more confident)
        rsi = technical_result.get("rsi", 50)
        rsi_conf = 1.0 - abs(rsi - 50) / 50  # 0-1, 1 if RSI=0 or 100, 0 if RSI=50
        confidence_factors.append(rsi_conf)

        # Momentum confidence (trend strength)
        momentum = abs(technical_result.get("momentum", 0.0))
        momentum_conf = min(momentum / 10, 1.0)
        confidence_factors.append(momentum_conf)

        # Volatility penalty
        if volatility_result.get("is_high_vol", False):
            confidence_factors.append(0.6)  # Confidence reduction
        else:
            confidence_factors.append(0.9)  # Good confidence

        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Weight for each factor
        total_conf = sum(w * c for w, c in zip(weights, confidence_factors))

        return min(max(total_conf, 0.1), 0.95)  # Between 10% and 95%

    def _generate_reasons(self, signal: str, arima_result: Dict,
                         technical_result: Dict, volatility_result: Dict) -> List[str]:
        """Generate signal reasons"""
        reasons = []

        # ARIMA
        arima_signal = arima_result.get("signal", 0)
        if arima_signal > 0:
            reasons.append("Positive ARIMA forecast")
        elif arima_signal < 0:
            reasons.append("Negative ARIMA forecast")

        # RSI
        rsi = technical_result.get("rsi", 50)
        if rsi < 30:
            reasons.append("RSI indicates oversold")
        elif rsi > 70:
            reasons.append("RSI indicates overbought")

        # Momentum
        momentum = technical_result.get("momentum", 0.0)
        if abs(momentum) > 2.0:
            if momentum > 0:
                reasons.append("Strong bullish trend")
            else:
                reasons.append("Strong bearish trend")

        # Volatility
        if volatility_result.get("is_high_vol", False):
            reasons.append("High market volatility - caution recommended")

        # Final message based on signal
        if signal == "BUY":
            reasons.append("Buy signals dominant")
        elif signal == "SELL":
            reasons.append("Sell signals dominant")
        else:
            reasons.append("Mixed signals - hold recommended")

        return reasons[:4]  # Limit to 4 reasons

    def _aggregate_signals(self, results: List[Dict]) -> Dict:
        """Aggregate signals from multiple symbols"""
        if not results:
            return {"decision": "HOLD", "confidence": 0.5, "score": 0.0, "reasons": ["No symbols analyzed"]}

        # Count signals
        buy_count = sum(1 for r in results if r.get("signal") == "BUY")
        sell_count = sum(1 for r in results if r.get("signal") == "SELL")
        hold_count = sum(1 for r in results if r.get("signal") == "HOLD")

        # Calculate average confidence
        avg_confidence = np.mean([r.get("confidence", 0.5) for r in results])

        # Aggregate score (average of normalized scores)
        scores = []
        for r in results:
            signal = r.get("signal", "HOLD")
            conf = r.get("confidence", 0.5)

            if signal == "BUY":
                scores.append(conf)
            elif signal == "SELL":
                scores.append(-conf)
            else:
                scores.append(0)

        avg_score = np.mean(scores) if scores else 0.0

        # Final decision
        if buy_count > sell_count and buy_count > hold_count:
            decision = "BUY"
            confidence = avg_confidence
            reasons = [f"{buy_count} buy signals out of {len(results)} symbols"]
        elif sell_count > buy_count and sell_count > hold_count:
            decision = "SELL"
            confidence = avg_confidence
            reasons = [f"{sell_count} sell signals out of {len(results)} symbols"]
        else:
            decision = "HOLD"
            confidence = 0.6  # Moderate confidence for HOLD
            reasons = [f"Mixed signals: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD"]

        return {
            "decision": decision,
            "confidence": float(confidence),
            "score": float(avg_score),
            "reasons": reasons
        }

    def quick_analysis(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Quick analysis for backward compatibility"""
        try:
            # This method is kept for compatibility
            # but now uses the new methods

            if len(data) < 10:
                return self._quick_analysis_fallback(symbol)

            # Run analyses
            technical_result = self._technical_analysis(data)
            volatility_result = self._volatility_analysis(data)

            # For ARIMA, we can do a simplified version
            arima_result = {"signal": 0, "forecast": 0.0}
            try:
                if len(data) >= 20:
                    prices = data['Close'].dropna()
                    returns = np.log(prices / prices.shift(1)).dropna()

                    # Simplified version without full fitting
                    mean_return = returns.mean()
                    arima_result["signal"] = 1 if mean_return > 0 else -1 if mean_return < 0 else 0
                    arima_result["forecast"] = float(mean_return)
            except:
                pass

            # Combine signals
            signal = self._combine_signals(arima_result, technical_result, volatility_result)
            confidence = self._calculate_confidence(arima_result, technical_result, volatility_result)

            return {
                'decision': signal,
                'action': f"{signal} - Quantitative analysis",
                'score': arima_result.get("forecast", 0.0),
                'confidence': confidence * 100,
                'reasons': self._generate_reasons(signal, arima_result, technical_result, volatility_result)
            }

        except Exception as e:
            print(f"⚠️ Quick analysis failed for {symbol}: {e}")
            return self._quick_analysis_fallback(symbol)

    def _quick_analysis_fallback(self, symbol: str) -> Dict:
        """Fallback when analysis fails"""
        return {
            'decision': 'HOLD',
            'action': 'HOLD - Insufficient data',
            'score': 0.0,
            'confidence': 40,
            'reasons': ['Insufficient data for quantitative analysis']
        }


# Adapt for orchestrator
class QuantModelerAgentAdapter:
    """Adapter for MultiAgentOrchestrator"""

    def __init__(self, use_cache: bool = True):
        self.quant_agent = TradingQuantAgent(use_cache=use_cache)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface"""
        return await self.quant_agent.run(context)


# Example usage
async def main_example():
    """Example usage of the quant agent"""

    # Create agent
    agent = TradingQuantAgent()

    # Example context
    context = {
        "market_data": {
            "AAPL": {"price": 185.50, "vol": 0.15},
            "SPY": {"price": 4700.00, "vol": 0.12}
        },
        "query": "Should I buy Apple stock?",
        "price_series": [180, 182, 184, 183, 185, 186, 185, 184, 186, 185.5],
        "risk_params": {"max_exposure": 1.0},
        "rag_summary": "Apple announces new products"
    }

    # Run analysis
    print("🔍 Running quantitative analysis...")
    result = await agent.run(context)

    # Display results
    print("\n📊 QUANTITATIVE ANALYSIS RESULTS:")
    print("=" * 50)
    print(f"Signal: {result.get('signal', 'N/A')}")
    print(f"Score: {result.get('score', 0):.2f}")
    print(f"Symbols analyzed: {', '.join(result.get('symbols_analyzed', []))}")

    if 'detailed_analysis' in result and result['detailed_analysis']:
        print("\n📈 Details by symbol:")
        for analysis in result['detailed_analysis']:
            print(f"\n  {analysis['symbol']}:")
            print(f"    Price: ${analysis['current_price']:.2f}")
            print(f"    Signal: {analysis['signal']}")
            print(f"    Confidence: {analysis['confidence']:.1%}")
            print(f"    RSI: {analysis['rsi']:.1f}")

    print(f"\n🤔 Reasoning: {' | '.join(result.get('reasoning', []))}")
    print(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")

if __name__ == "__main__":
    # Run example
    asyncio.run(main_example())
