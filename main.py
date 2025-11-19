"""
BTC Scalper Bot - Main Entry Point
Production-ready trading bot with Telegram integration.
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
log_file = os.getenv('LOG_FILE', 'logs/trading_bot.log')

# Create logs directory
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import core modules
from core import BybitAPI, TradingBot, TelegramBot


class BotManager:
    """
    Main bot manager that orchestrates trading and Telegram bots.
    """

    def __init__(self):
        """Initialize bot manager."""
        self.trading_bot = None
        self.telegram_bot = None
        self.is_running = False

        # Load configuration
        self.load_config()

        # Initialize bots
        self.init_api()
        self.init_trading_bot()
        self.init_telegram_bot()

    def load_config(self):
        """Load configuration from environment variables."""
        # Trading mode
        self.trading_mode = os.getenv('TRADING_MODE', 'paper')

        # Bybit API
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.api_url = os.getenv('BYBIT_API_URL', 'https://api.bybit.com')

        # Telegram
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_enabled = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'

        # Trading config
        self.symbol = os.getenv('SYMBOL', 'BTCUSDT')
        self.timeframe = os.getenv('TIMEFRAME', '15')
        self.config_path = os.getenv('CONFIG_PATH', 'config_ultra_optimized_FINAL.yaml')
        self.model_path = os.getenv('MODEL_PATH', 'storage/models/ultra_scalper_btcusdt_365d.pkl')

        # Risk management
        self.max_position_size_usd = float(os.getenv('MAX_POSITION_SIZE_USD', '1000.0'))

        # Paper trading
        self.paper_balance = float(os.getenv('PAPER_INITIAL_BALANCE', '10000.0'))
        self.paper_slippage = float(os.getenv('PAPER_SLIPPAGE_PCT', '0.05'))
        self.paper_fee = float(os.getenv('PAPER_FEE_PCT', '0.055'))

        # Check interval
        self.check_interval = int(os.getenv('CHECK_INTERVAL_SECONDS', '60'))

        # Validation
        if not self.api_key or not self.api_secret:
            raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

        if self.telegram_enabled and (not self.telegram_token or not self.telegram_chat_id):
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

        logger.info(f"✅ Configuration loaded - Mode: {self.trading_mode.upper()}")

    def init_api(self):
        """Initialize Bybit API."""
        self.api = BybitAPI(
            api_key=self.api_key,
            api_secret=self.api_secret,
            mode=self.trading_mode,
            api_url=self.api_url,
            paper_balance=self.paper_balance,
            slippage_pct=self.paper_slippage,
            fee_pct=self.paper_fee
        )

        logger.info(f"✅ Bybit API initialized - Mode: {self.trading_mode}")

    def init_trading_bot(self):
        """Initialize trading bot."""
        self.trading_bot = TradingBot(
            api=self.api,
            config_path=self.config_path,
            model_path=self.model_path,
            symbol=self.symbol,
            timeframe=self.timeframe,
            max_position_size_usd=self.max_position_size_usd
        )

        logger.info(f"✅ Trading bot initialized")

    def init_telegram_bot(self):
        """Initialize Telegram bot."""
        if not self.telegram_enabled:
            logger.info("⏭️  Telegram disabled")
            return

        self.telegram_bot = TelegramBot(
            token=self.telegram_token,
            chat_id=self.telegram_chat_id,
            trading_bot=self.trading_bot
        )

        logger.info(f"✅ Telegram bot initialized")

    async def send_startup_message(self):
        """Send startup message via Telegram."""
        if not self.telegram_enabled or not self.telegram_bot:
            return

        message = (
            f"🤖 *BTC Scalper Bot Started*\n\n"
            f"Mode: {self.trading_mode.upper()}\n"
            f"Symbol: {self.symbol}\n"
            f"Timeframe: {self.timeframe}m\n"
            f"Max Position: ${self.max_position_size_usd:,.0f}\n\n"
            f"Use /help to see available commands."
        )

        try:
            await self.telegram_bot.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send startup message: {e}")

    async def run_trading_loop(self):
        """Run the main trading loop."""
        logger.info("🚀 Starting trading loop...")

        # Start trading bot
        self.trading_bot.start()
        self.is_running = True

        # Send startup message
        await self.send_startup_message()

        last_cycle_time = datetime.now()

        while self.is_running and self.trading_bot.is_running:
            try:
                # Run trading cycle
                self.trading_bot.run_cycle()

                # Log cycle completion
                cycle_duration = (datetime.now() - last_cycle_time).total_seconds()
                logger.debug(f"✅ Cycle completed in {cycle_duration:.1f}s")

                last_cycle_time = datetime.now()

                # Wait for next cycle
                await asyncio.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("⚠️  Keyboard interrupt received")
                break

            except Exception as e:
                logger.error(f"❌ Error in trading loop: {e}")
                await asyncio.sleep(self.check_interval)

        logger.info("🛑 Trading loop stopped")

    async def run(self):
        """Run the bot manager."""
        try:
            logger.info("=" * 80)
            logger.info("🚀 BTC SCALPER BOT STARTING")
            logger.info("=" * 80)

            if self.telegram_enabled:
                # Run both trading loop and Telegram bot concurrently
                telegram_app = self.telegram_bot.start_async()

                async with telegram_app:
                    await telegram_app.initialize()
                    await telegram_app.start()

                    # Run trading loop
                    await self.run_trading_loop()

                    await telegram_app.stop()
            else:
                # Run only trading loop
                await self.run_trading_loop()

        except KeyboardInterrupt:
            logger.info("⚠️  Bot stopped by user")

        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise

        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the bot gracefully."""
        logger.info("🛑 Shutting down...")

        if self.trading_bot:
            self.trading_bot.stop()

        # Print final stats
        if self.api.mode == 'paper':
            stats = self.api.get_performance_stats()

            if stats:
                logger.info("=" * 80)
                logger.info("📊 FINAL STATISTICS")
                logger.info("=" * 80)
                logger.info(f"Total Trades: {stats['total_trades']}")
                logger.info(f"Win Rate: {stats['win_rate']:.2f}%")
                logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
                logger.info(f"ROI: {stats['roi']:.2f}%")
                logger.info(f"Final Balance: ${stats['current_balance']:,.2f}")
                logger.info("=" * 80)

        logger.info("✅ Shutdown complete")


def main():
    """Main entry point."""
    try:
        # Create bot manager
        manager = BotManager()

        # Run the bot
        asyncio.run(manager.run())

    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
