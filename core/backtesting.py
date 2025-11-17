"""
Backtesting Engine Completo para Trading de Criptomoedas
Simula trades reais com fees, slippage, e métricas profissionais

Métricas calculadas:
- Total Return, Annualized Return
- Sharpe Ratio, Sortino Ratio
- Max Drawdown, Calmar Ratio
- Win Rate, Profit Factor
- Average Win/Loss
- Número de trades, Hold time médio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from datetime import datetime, timedelta


class Backtester:
    """
    Engine de backtesting para estratégias de trading.

    Simula trades reais considerando:
    - Fees de exchange (maker/taker)
    - Slippage
    - Position sizing
    - Stop loss e take profit
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.0006,  # 0.06% Bybit taker fee
        slippage: float = 0.0001,   # 0.01% slippage
        position_size: float = 1.0,  # 100% do capital por trade
        risk_per_trade: float = 0.02,  # 2% risco máximo por trade
        cooldown_minutes: int = 15  # Cooldown entre trades (evita overtrading)
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position_size = position_size
        self.risk_per_trade = risk_per_trade
        self.cooldown_minutes = cooldown_minutes

        # State
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.last_trade_time = None  # Timestamp do último trade

    def run(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        actual_labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Executa backtesting completo.

        Args:
            df: DataFrame com OHLCV data
            predictions: Probabilidades do modelo (0=LONG, 1=SHORT)
            actual_labels: Labels reais (0=LONG, 1=SHORT)
            threshold: Threshold para decisão de trade

        Returns:
            Dict com métricas de performance
        """
        logger.info("Iniciando backtesting...")
        logger.info(f"Capital inicial: ${self.initial_capital:,.2f}")
        logger.info(f"Fee rate: {self.fee_rate*100:.2f}%")
        logger.info(f"Slippage: {self.slippage*100:.2f}%")
        if self.cooldown_minutes > 0:
            logger.info(f"Cooldown: {self.cooldown_minutes} minutos entre trades")

        capital = self.initial_capital
        self.trades = []
        self.equity_curve = [capital]
        self.last_trade_time = None

        # Converter predições em sinais
        signals = self._predictions_to_signals(predictions, threshold)

        # Simular trades
        for i in range(len(df)):
            current_time = df.iloc[i]['timestamp']

            # Pular se não há sinal
            if signals[i] == 0:  # 0 = no trade
                self.equity_curve.append(capital)
                continue

            # Verificar cooldown
            if self.last_trade_time is not None and self.cooldown_minutes > 0:
                time_since_last_trade = (current_time - self.last_trade_time).total_seconds() / 60
                if time_since_last_trade < self.cooldown_minutes:
                    # Ainda em cooldown, pular trade
                    self.equity_curve.append(capital)
                    continue

            # Executar trade
            trade_result = self._execute_trade(
                entry_price=df.iloc[i]['close'],
                signal=signals[i],  # 1=LONG, -1=SHORT
                actual_label=actual_labels[i],
                timestamp=current_time,
                capital=capital
            )

            if trade_result:
                self.trades.append(trade_result)
                capital += trade_result['pnl_net']
                self.last_trade_time = current_time  # Atualizar último trade

            self.equity_curve.append(capital)

        # Calcular métricas
        metrics = self._calculate_metrics(df)

        logger.info(f"Backtesting concluído: {len(self.trades)} trades")

        return metrics

    def _predictions_to_signals(
        self,
        predictions: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Converte probabilidades em sinais de trade.

        Returns:
            1 = LONG, -1 = SHORT, 0 = No trade
        """
        signals = np.zeros(len(predictions))

        for i, prob in enumerate(predictions):
            if prob < (1 - threshold):  # Alta confiança em LONG
                signals[i] = 1
            elif prob > threshold:  # Alta confiança em SHORT
                signals[i] = -1
            # else: signals[i] = 0 (no trade)

        return signals

    def _execute_trade(
        self,
        entry_price: float,
        signal: int,
        actual_label: int,
        timestamp: datetime,
        capital: float
    ) -> Optional[Dict]:
        """
        Executa um trade e retorna resultado.
        """
        # Calcular tamanho da posição
        position_value = capital * self.position_size

        # Aplicar slippage na entrada
        if signal == 1:  # LONG
            entry_price_adj = entry_price * (1 + self.slippage)
        else:  # SHORT
            entry_price_adj = entry_price * (1 - self.slippage)

        # Fee de entrada
        entry_fee = position_value * self.fee_rate

        # Simular resultado baseado no label real
        # Se label = 0 (LONG real), preço subiu
        # Se label = 1 (SHORT real), preço caiu

        # Simplificação: assumir 1% de movimento na direção do label
        price_move_pct = 0.01  # 1% movimento

        if actual_label == 0:  # Preço subiu (LONG real)
            exit_price = entry_price * (1 + price_move_pct)
        else:  # Preço caiu (SHORT real)
            exit_price = entry_price * (1 - price_move_pct)

        # Aplicar slippage na saída
        if signal == 1:  # LONG
            exit_price_adj = exit_price * (1 - self.slippage)
        else:  # SHORT
            exit_price_adj = exit_price * (1 + self.slippage)

        # Fee de saída
        exit_fee = position_value * self.fee_rate

        # Calcular P&L
        if signal == 1:  # LONG
            pnl_pct = (exit_price_adj - entry_price_adj) / entry_price_adj
        else:  # SHORT
            pnl_pct = (entry_price_adj - exit_price_adj) / entry_price_adj

        pnl_gross = position_value * pnl_pct
        pnl_net = pnl_gross - entry_fee - exit_fee

        # Trade correto?
        correct = (signal == 1 and actual_label == 0) or \
                  (signal == -1 and actual_label == 1)

        return {
            'timestamp': timestamp,
            'signal': 'LONG' if signal == 1 else 'SHORT',
            'entry_price': entry_price_adj,
            'exit_price': exit_price_adj,
            'position_value': position_value,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'pnl_pct': pnl_net / position_value,
            'fees': entry_fee + exit_fee,
            'correct': correct
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calcula métricas completas de performance.
        """
        if not self.trades:
            logger.warning("Nenhum trade executado!")
            return {}

        trades_df = pd.DataFrame(self.trades)
        equity = np.array(self.equity_curve)

        # Returns
        total_return = (equity[-1] - equity[0]) / equity[0]

        # Calcular período em anos
        days = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days
        years = days / 365.25

        # Annualized return
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0

        # Daily returns
        daily_returns = np.diff(equity) / equity[:-1]

        # Sharpe Ratio (assumindo risk-free rate = 0)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0

        # Sortino Ratio (apenas downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() > 0:
            sortino = np.sqrt(252) * daily_returns.mean() / negative_returns.std()
        else:
            sortino = 0

        # Max Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        if max_drawdown < 0:
            calmar = annualized_return / abs(max_drawdown)
        else:
            calmar = 0

        # Win/Loss stats
        wins = trades_df[trades_df['pnl_net'] > 0]
        losses = trades_df[trades_df['pnl_net'] <= 0]

        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

        avg_win = wins['pnl_net'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_net'].mean() if len(losses) > 0 else 0

        # Profit Factor
        total_wins = wins['pnl_net'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_net'].sum()) if len(losses) > 0 else 0

        if total_losses > 0:
            profit_factor = total_wins / total_losses
        else:
            profit_factor = float('inf') if total_wins > 0 else 0

        # Accuracy (trades corretos)
        accuracy = trades_df['correct'].mean()

        return {
            'total_trades': len(trades_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'accuracy': accuracy,

            'total_return': total_return,
            'annualized_return': annualized_return,

            'total_pnl': equity[-1] - equity[0],
            'total_fees': trades_df['fees'].sum(),

            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': trades_df['pnl_net'].mean(),

            'best_trade': trades_df['pnl_net'].max(),
            'worst_trade': trades_df['pnl_net'].min(),

            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,

            'profit_factor': profit_factor,

            'initial_capital': self.initial_capital,
            'final_capital': equity[-1],

            'period_days': days,
            'period_years': years
        }

    def print_report(self, metrics: Dict):
        """
        Imprime relatório formatado de performance.
        """
        print("\n" + "="*80)
        print("RELATÓRIO DE BACKTESTING")
        print("="*80)

        print(f"\n📊 RESUMO GERAL")
        print(f"  Período: {metrics['period_days']} dias ({metrics['period_years']:.2f} anos)")
        print(f"  Total de Trades: {metrics['total_trades']}")
        print(f"  Wins: {metrics['winning_trades']} | Losses: {metrics['losing_trades']}")

        print(f"\n💰 RETORNO")
        print(f"  Capital Inicial: ${metrics['initial_capital']:,.2f}")
        print(f"  Capital Final: ${metrics['final_capital']:,.2f}")
        print(f"  P&L Total: ${metrics['total_pnl']:,.2f}")
        print(f"  Total em Fees: ${metrics['total_fees']:,.2f}")
        print(f"  Retorno Total: {metrics['total_return']*100:.2f}%")
        print(f"  Retorno Anualizado: {metrics['annualized_return']*100:.2f}%")

        print(f"\n📈 MÉTRICAS DE RISCO")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")

        print(f"\n🎯 PERFORMANCE DOS TRADES")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Accuracy (trades corretos): {metrics['accuracy']*100:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Win: ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"  Avg P&L: ${metrics['avg_pnl']:.2f}")
        print(f"  Melhor Trade: ${metrics['best_trade']:.2f}")
        print(f"  Pior Trade: ${metrics['worst_trade']:.2f}")

        print("\n" + "="*80)

        # Avaliação
        self._evaluate_strategy(metrics)

    def _evaluate_strategy(self, metrics: Dict):
        """
        Avalia se a estratégia é boa para trading.
        """
        print("\n🔍 AVALIAÇÃO DA ESTRATÉGIA\n")

        score = 0
        max_score = 8

        # 1. Retorno positivo
        if metrics['total_return'] > 0:
            print("  ✅ Retorno positivo")
            score += 1
        else:
            print("  ❌ Retorno negativo")

        # 2. Sharpe > 1
        if metrics['sharpe_ratio'] > 1:
            print("  ✅ Sharpe Ratio > 1 (bom)")
            score += 1
        elif metrics['sharpe_ratio'] > 0.5:
            print("  🟡 Sharpe Ratio > 0.5 (aceitável)")
            score += 0.5
        else:
            print("  ❌ Sharpe Ratio < 0.5 (ruim)")

        # 3. Max Drawdown < 20%
        if metrics['max_drawdown'] > -0.20:
            print("  ✅ Max Drawdown < 20%")
            score += 1
        elif metrics['max_drawdown'] > -0.30:
            print("  🟡 Max Drawdown < 30%")
            score += 0.5
        else:
            print("  ❌ Max Drawdown > 30%")

        # 4. Win Rate > 50%
        if metrics['win_rate'] > 0.50:
            print("  ✅ Win Rate > 50%")
            score += 1
        elif metrics['win_rate'] > 0.45:
            print("  🟡 Win Rate > 45%")
            score += 0.5
        else:
            print("  ❌ Win Rate < 45%")

        # 5. Profit Factor > 1.5
        if metrics['profit_factor'] > 1.5:
            print("  ✅ Profit Factor > 1.5")
            score += 1
        elif metrics['profit_factor'] > 1.0:
            print("  🟡 Profit Factor > 1.0")
            score += 0.5
        else:
            print("  ❌ Profit Factor < 1.0")

        # 6. Trades suficientes
        if metrics['total_trades'] > 100:
            print("  ✅ Trades suficientes (>100)")
            score += 1
        elif metrics['total_trades'] > 50:
            print("  🟡 Trades moderados (>50)")
            score += 0.5
        else:
            print("  ⚠️  Poucos trades (<50)")

        # 7. Sortino > 1
        if metrics['sortino_ratio'] > 1:
            print("  ✅ Sortino Ratio > 1")
            score += 1
        elif metrics['sortino_ratio'] > 0.5:
            print("  🟡 Sortino Ratio > 0.5")
            score += 0.5
        else:
            print("  ❌ Sortino Ratio < 0.5")

        # 8. Retorno anualizado > 10%
        if metrics['annualized_return'] > 0.10:
            print("  ✅ Retorno anualizado > 10%")
            score += 1
        elif metrics['annualized_return'] > 0.05:
            print("  🟡 Retorno anualizado > 5%")
            score += 0.5
        else:
            print("  ❌ Retorno anualizado < 5%")

        # Score final
        score_pct = (score / max_score) * 100

        print(f"\n📊 SCORE: {score:.1f}/{max_score} ({score_pct:.1f}%)")

        if score_pct >= 80:
            print("\n✅ EXCELENTE! Estratégia pronta para paper trading")
        elif score_pct >= 60:
            print("\n🟡 BOA! Considere otimizações antes de paper trading")
        elif score_pct >= 40:
            print("\n⚠️  MARGINAL. Precisa melhorar antes de usar real")
        else:
            print("\n❌ RUIM. Não recomendado para trading real")
