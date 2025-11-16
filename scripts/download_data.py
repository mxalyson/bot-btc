#!/usr/bin/env python3
"""
Script CLI para download de dados históricos da Bybit.

Uso:
    python scripts/download_data.py --symbol BTCUSDT --timeframe 5m --days 180
    python scripts/download_data.py --all  # Baixa todos os símbolos do config
"""

import argparse
import sys
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import load_config, setup_logging, print_section
from core.data_loader import BybitDataLoader
from loguru import logger


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description='Download de dados históricos da Bybit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Baixar BTC em timeframe de 5m, últimos 180 dias
  python scripts/download_data.py --symbol BTCUSDT --timeframe 5m --days 180

  # Baixar múltiplos timeframes
  python scripts/download_data.py --symbol ETHUSDT --timeframe 1m 5m 15m --days 90

  # Baixar todos os símbolos configurados em config.yaml
  python scripts/download_data.py --all

  # Usar arquivo de configuração customizado
  python scripts/download_data.py --config my_config.yaml --all
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Caminho para arquivo de configuração (default: config/config.yaml)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        help='Símbolo do ativo (ex: BTCUSDT, ETHUSDT)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        nargs='+',
        help='Timeframe(s) (ex: 1m 5m 15m)'
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Número de dias de histórico'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Baixar todos os símbolos e timeframes do config'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Data inicial (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='Data final (YYYY-MM-DD)'
    )

    return parser.parse_args()


def download_single(
    loader: BybitDataLoader,
    symbol: str,
    timeframe: str,
    days: int = None,
    start_date: str = None,
    end_date: str = None
):
    """
    Baixa dados para um único símbolo/timeframe.
    """
    print_section(f"Download: {symbol} | {timeframe}")

    try:
        # Download
        if start_date and end_date:
            df = loader.download_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
        else:
            df = loader.download_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                lookback_days=days
            )

        if df.empty:
            logger.error(f"✗ Nenhum dado baixado para {symbol} {timeframe}")
            return False

        # Salvar
        file_path = loader.save_data(df, symbol, timeframe, data_type="raw")
        logger.info(f"✓ Sucesso! {len(df)} candles salvos em {file_path}")

        return True

    except Exception as e:
        logger.error(f"✗ Erro ao baixar {symbol} {timeframe}: {e}")
        return False


def download_all(loader: BybitDataLoader, config: dict):
    """
    Baixa todos os símbolos e timeframes configurados.
    """
    print_section("Download de Todos os Símbolos")

    symbols = config['data']['symbols']
    timeframes = config['data']['timeframes']
    lookback_days = config['data']['lookback_days']

    logger.info(f"Símbolos: {symbols}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Período: {lookback_days} dias")

    total = len(symbols) * len(timeframes)
    success_count = 0
    fail_count = 0

    for i, symbol in enumerate(symbols, 1):
        for j, timeframe in enumerate(timeframes, 1):
            current = (i - 1) * len(timeframes) + j

            logger.info(f"\n[{current}/{total}] Processando {symbol} | {timeframe}")

            if download_single(loader, symbol, timeframe, days=lookback_days):
                success_count += 1
            else:
                fail_count += 1

    # Resumo
    print_section("Resumo do Download")
    logger.info(f"Total de combinações: {total}")
    logger.info(f"Sucessos: {success_count}")
    logger.info(f"Falhas: {fail_count}")

    if fail_count > 0:
        logger.warning(f"⚠ {fail_count} downloads falharam")
    else:
        logger.info("✓ Todos os downloads concluídos com sucesso!")


def main():
    """Função principal."""
    args = parse_args()

    # Carregar configuração
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Erro: Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)

    # Setup logging
    setup_logging(config)

    print_section("Bybit Data Downloader")
    logger.info(f"Configuração carregada: {args.config}")

    # Inicializar loader
    loader = BybitDataLoader(config)

    # Modo: baixar todos ou específico
    if args.all:
        download_all(loader, config)
    else:
        # Validar argumentos
        if not args.symbol:
            logger.error("Erro: --symbol é obrigatório quando não usando --all")
            sys.exit(1)

        if not args.timeframe:
            logger.error("Erro: --timeframe é obrigatório quando não usando --all")
            sys.exit(1)

        # Determinar período
        if args.start_date and args.end_date:
            days = None
        elif args.days:
            days = args.days
        else:
            # Usar padrão do config
            days = config['data']['lookback_days']
            logger.info(f"Usando período padrão do config: {days} dias")

        # Download para cada timeframe especificado
        success_count = 0
        for timeframe in args.timeframe:
            if download_single(
                loader,
                args.symbol,
                timeframe,
                days=days,
                start_date=args.start_date,
                end_date=args.end_date
            ):
                success_count += 1

        # Resumo
        total = len(args.timeframe)
        logger.info(f"\n✓ {success_count}/{total} downloads concluídos")

    logger.info("\nFinalizado!")


if __name__ == "__main__":
    main()
