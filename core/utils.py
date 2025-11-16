"""
Funções utilitárias para o pipeline de ML.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import sys


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configura o sistema de logging usando loguru.

    Args:
        config: Dicionário de configuração com seção 'logging'
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_dir = log_config.get('log_dir', 'logs')
    log_file = log_config.get('log_file', 'training.log')

    # Criar diretório de logs se não existir
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remover handlers padrão
    logger.remove()

    # Adicionar handler para console
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # Adicionar handler para arquivo
    log_path = Path(log_dir) / log_file
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level=log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip"
    )

    logger.info(f"Logging configurado - Nível: {log_level}")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carrega o arquivo de configuração YAML.

    Args:
        config_path: Caminho para o arquivo config.yaml

    Returns:
        Dicionário com as configurações
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuração carregada de: {config_path}")
    return config


def ensure_dir(directory: str) -> None:
    """
    Garante que um diretório existe, criando-o se necessário.

    Args:
        directory: Caminho do diretório
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_data_path(symbol: str, timeframe: str, data_type: str = "raw") -> Path:
    """
    Retorna o caminho para um arquivo de dados.

    Args:
        symbol: Símbolo do ativo (ex: BTCUSDT)
        timeframe: Timeframe (ex: 1m, 5m)
        data_type: Tipo de dados - "raw" ou "processed"

    Returns:
        Path object com o caminho do arquivo
    """
    base_dir = f"data/{data_type}"
    ensure_dir(base_dir)
    filename = f"{symbol}_{timeframe}.csv"
    return Path(base_dir) / filename


def save_model_metadata(model_path: Path, metadata: Dict[str, Any]) -> None:
    """
    Salva metadados do modelo em arquivo YAML.

    Args:
        model_path: Caminho do modelo
        metadata: Dicionário com metadados
    """
    metadata_path = model_path.with_suffix('.yaml')

    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    logger.info(f"Metadados salvos em: {metadata_path}")


def load_model_metadata(model_path: Path) -> Dict[str, Any]:
    """
    Carrega metadados do modelo.

    Args:
        model_path: Caminho do modelo

    Returns:
        Dicionário com metadados
    """
    metadata_path = model_path.with_suffix('.yaml')

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = yaml.safe_load(f)

    return metadata


def format_number(num: float, decimals: int = 2) -> str:
    """
    Formata um número para exibição.

    Args:
        num: Número para formatar
        decimals: Número de casas decimais

    Returns:
        String formatada
    """
    return f"{num:.{decimals}f}"


def format_percentage(num: float, decimals: int = 2) -> str:
    """
    Formata um número como porcentagem.

    Args:
        num: Número para formatar (0.15 = 15%)
        decimals: Número de casas decimais

    Returns:
        String formatada com símbolo %
    """
    return f"{num * 100:.{decimals}f}%"


def print_section(title: str, width: int = 80) -> None:
    """
    Imprime um cabeçalho de seção formatado.

    Args:
        title: Título da seção
        width: Largura total do cabeçalho
    """
    logger.info("=" * width)
    logger.info(f"{title.center(width)}")
    logger.info("=" * width)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Valida a configuração para garantir que todas as seções necessárias existem.

    Args:
        config: Dicionário de configuração

    Raises:
        ValueError: Se alguma seção obrigatória estiver faltando
    """
    required_sections = ['data', 'features', 'labeling', 'dataset', 'model', 'training']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Seção obrigatória '{section}' não encontrada na configuração")

    # Validar símbolos
    if not config['data'].get('symbols'):
        raise ValueError("Lista de símbolos vazia em data.symbols")

    # Validar timeframes
    if not config['data'].get('timeframes'):
        raise ValueError("Lista de timeframes vazia em data.timeframes")

    # Validar divisão do dataset
    splits = [
        config['dataset']['train_split'],
        config['dataset']['val_split'],
        config['dataset']['test_split']
    ]

    if abs(sum(splits) - 1.0) > 0.01:
        raise ValueError(f"Soma das divisões do dataset deve ser 1.0, obteve: {sum(splits)}")

    logger.info("Configuração validada com sucesso")


if __name__ == "__main__":
    # Teste das funções
    config = load_config()
    setup_logging(config)
    validate_config(config)
    print_section("Teste do módulo utils.py")
    logger.info("Todas as funções utilitárias foram testadas")
