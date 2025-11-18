"""
🎯 PURGED K-FOLD CROSS-VALIDATION
Implementação de Purged K-Fold CV para time series financeiras

Baseado em "Advances in Financial Machine Learning" por Marcos López de Prado

Diferenças do K-Fold tradicional:
- Remove samples entre train/test (embargo/purge)
- Previne vazamento de informação temporal
- Mais rigoroso que Walk-Forward

Uso:
    from purged_kfold import PurgedKFold, CombinatorialPurgedCV

    cv = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=24))
    for train_idx, test_idx in cv.split(X, y):
        # Treina e testa
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation para time series

    Remove samples entre train e test sets para prevenir leakage temporal

    Parâmetros:
    -----------
    n_splits : int
        Número de folds
    embargo_td : pd.Timedelta
        Tempo de embargo após cada test set
    purge_td : pd.Timedelta, opcional
        Tempo de purge antes de cada test set (padrão: mesmo que embargo)
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_td: pd.Timedelta = pd.Timedelta(hours=1),
        purge_td: Optional[pd.Timedelta] = None
    ):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td if purge_td is not None else embargo_td

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera índices para train/test splits com purging e embargo

        Parameters:
        -----------
        X : array-like
            Features (deve ter index temporal se for DataFrame)
        y : array-like, opcional
            Target
        groups : array-like, opcional
            Não usado, mantido para compatibilidade

        Yields:
        -------
        train : ndarray
            Índices de treino
        test : ndarray
            Índices de teste
        """
        if isinstance(X, pd.DataFrame):
            indices = np.arange(len(X))
            timestamps = X.index
        elif isinstance(X, pd.Series):
            indices = np.arange(len(X))
            timestamps = X.index
        else:
            # Se não tem timestamps, usa índices sequenciais
            indices = np.arange(len(X))
            timestamps = pd.date_range(start='2020-01-01', periods=len(X), freq='1h')

        # Valida n_splits
        n_samples = len(indices)
        if self.n_splits > n_samples:
            raise ValueError(f"n_splits={self.n_splits} não pode ser maior que n_samples={n_samples}")

        # Tamanho de cada test fold
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Define test set
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples

            test_indices = indices[test_start:test_end]

            # Define train set (tudo exceto test)
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])

            # PURGE: Remove samples antes do test que podem vazar informação
            if len(train_indices) > 0 and len(test_indices) > 0:
                test_start_time = timestamps[test_indices[0]]

                # Remove do train todas as amostras dentro de purge_td antes do test
                purge_cutoff = test_start_time - self.purge_td

                # Índices do train que devem ser mantidos
                train_times = timestamps[train_indices]
                keep_train_mask = (train_times < purge_cutoff) | (train_times > timestamps[test_indices[-1]] + self.embargo_td)

                train_indices = train_indices[keep_train_mask]

            # EMBARGO: Remove samples depois do test (já incluído no keep_train_mask acima)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Retorna número de splits"""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)

    Testa TODAS combinações possíveis de train/test splits
    Mais rigoroso que Purged K-Fold, detecta overfitting de forma mais robusta

    Referência: López de Prado (2018), "Advances in Financial ML", Chapter 12
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        embargo_td: pd.Timedelta = pd.Timedelta(hours=1),
        purge_td: Optional[pd.Timedelta] = None
    ):
        """
        Args:
            n_splits: Número total de splits
            n_test_splits: Quantos splits usar para teste em cada combinação
            embargo_td: Tempo de embargo
            purge_td: Tempo de purge
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td if purge_td is not None else embargo_td

    def _get_combinations(self, n_splits: int, n_test: int):
        """Gera todas combinações de test splits"""
        from itertools import combinations
        return list(combinations(range(n_splits), n_test))

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Gera todas combinações de train/test splits

        Yields:
            train_indices, test_indices
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            indices = np.arange(len(X))
            timestamps = X.index
        else:
            indices = np.arange(len(X))
            timestamps = pd.date_range(start='2020-01-01', periods=len(X), freq='1h')

        n_samples = len(indices)
        split_size = n_samples // self.n_splits

        # Divide em splits
        splits = []
        for i in range(self.n_splits):
            start = i * split_size
            end = (i + 1) * split_size if i < self.n_splits - 1 else n_samples
            splits.append(indices[start:end])

        # Gera todas combinações
        test_combinations = self._get_combinations(self.n_splits, self.n_test_splits)

        for test_split_ids in test_combinations:
            # Test set = união dos splits selecionados
            test_indices = np.concatenate([splits[i] for i in test_split_ids])

            # Train set = todos os outros splits
            train_split_ids = [i for i in range(self.n_splits) if i not in test_split_ids]
            train_indices = np.concatenate([splits[i] for i in train_split_ids])

            # Apply purging and embargo
            if len(train_indices) > 0 and len(test_indices) > 0:
                test_times = timestamps[test_indices]
                train_times = timestamps[train_indices]

                # Purge: remover amostras de treino próximas ao test
                test_start = test_times.min()
                test_end = test_times.max()

                purge_start = test_start - self.purge_td
                embargo_end = test_end + self.embargo_td

                # Manter apenas amostras de treino fora da zona de purge/embargo
                keep_mask = (train_times < purge_start) | (train_times > embargo_end)
                train_indices = train_indices[keep_mask]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Retorna número de combinações"""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


class PurgedWalkForward:
    """
    Walk-Forward com Purging e Embargo

    Validação temporal com proteção contra leakage
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        embargo_td: pd.Timedelta = pd.Timedelta(hours=1),
        expanding_window: bool = False
    ):
        """
        Args:
            n_splits: Número de folds
            train_size: Tamanho fixo do treino (None = usa todo histórico disponível)
            test_size: Tamanho do teste
            embargo_td: Tempo de embargo
            expanding_window: Se True, janela de treino expande; se False, é fixa
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.embargo_td = embargo_td
        self.expanding_window = expanding_window

    def split(
        self,
        X,
        y=None,
        groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Gera walk-forward splits com embargo"""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            indices = np.arange(len(X))
            timestamps = X.index
        else:
            indices = np.arange(len(X))
            timestamps = pd.date_range(start='2020-01-01', periods=len(X), freq='1h')

        n_samples = len(indices)

        # Determina tamanho do test se não especificado
        test_size = self.test_size if self.test_size else n_samples // (self.n_splits + 1)

        # Calcula embargo em número de samples
        if len(timestamps) > 1:
            avg_timedelta = (timestamps[-1] - timestamps[0]) / len(timestamps)
            embargo_samples = int(self.embargo_td / avg_timedelta)
        else:
            embargo_samples = 0

        for i in range(self.n_splits):
            # Test set
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            test_indices = indices[test_start:test_end]

            # Train set
            if self.expanding_window:
                # Expanding: usa todo histórico até o test
                train_start = 0
            else:
                # Rolling: usa janela fixa
                if self.train_size:
                    train_start = max(0, test_start - self.train_size - embargo_samples)
                else:
                    train_start = 0

            train_end = test_start - embargo_samples  # Aplica embargo
            train_indices = indices[train_start:train_end]

            if len(train_indices) > 0:
                yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ============================================================================
# UTILITIES
# ============================================================================

def plot_cv_indices(cv, X, y, ax, lw=10):
    """
    Visualiza os splits de CV

    Args:
        cv: Cross-validator
        X: Features
        y: Target
        ax: Matplotlib axis
        lw: Largura da linha
    """
    n_splits = cv.get_n_splits(X, y)

    # Gera splits
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Plota train
        ax.fill_between(tr, ii, ii + 0.4, facecolor='blue', alpha=0.5, label='Train' if ii == 0 else '')
        # Plota test
        ax.fill_between(tt, ii + 0.4, ii + 0.8, facecolor='red', alpha=0.5, label='Test' if ii == 0 else '')

    ax.set_ylabel('CV Iteration')
    ax.set_xlabel('Sample Index')
    ax.legend(loc='upper left')
    ax.set_ylim([n_splits, -0.2])
    ax.set_xlim([0, len(X)])


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """Demonstração dos cross-validators"""

    print("🎯 PURGED K-FOLD CROSS-VALIDATION - DEMO")
    print("="*80)

    # Cria dados de exemplo com timestamps
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    }, index=dates)
    y = pd.Series(np.random.randint(0, 2, 1000), index=dates)

    # Teste 1: Purged K-Fold
    print("\n📊 TEST 1: PURGED K-FOLD")
    print("-"*80)
    pkf = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=1))

    print(f"Número de splits: {pkf.get_n_splits(X)}")
    print()

    for fold, (train_idx, test_idx) in enumerate(pkf.split(X, y)):
        train_start = X.index[train_idx[0]] if len(train_idx) > 0 else None
        train_end = X.index[train_idx[-1]] if len(train_idx) > 0 else None
        test_start = X.index[test_idx[0]]
        test_end = X.index[test_idx[-1]]

        print(f"Fold {fold + 1}:")
        print(f"  Train: {len(train_idx):4d} samples | {train_start} to {train_end}")
        print(f"  Test:  {len(test_idx):4d} samples | {test_start} to {test_end}")

        # Verifica gap
        if len(train_idx) > 0:
            last_train = X.index[train_idx[-1]]
            first_test = X.index[test_idx[0]]
            gap = first_test - last_train
            print(f"  Gap (purge): {gap}")
        print()

    # Teste 2: Combinatorial Purged CV
    print("\n📊 TEST 2: COMBINATORIAL PURGED CV")
    print("-"*80)
    cpcv = CombinatorialPurgedCV(
        n_splits=5,
        n_test_splits=1,
        embargo_td=pd.Timedelta(hours=1)
    )

    n_combinations = cpcv.get_n_splits(X)
    print(f"Número de combinações: {n_combinations}")
    print("\nPrimeiras 3 combinações:")

    for i, (train_idx, test_idx) in enumerate(cpcv.split(X, y)):
        if i >= 3:
            break

        print(f"\nCombinação {i + 1}:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Test:  {len(test_idx)} samples")

    # Teste 3: Purged Walk-Forward
    print("\n📊 TEST 3: PURGED WALK-FORWARD")
    print("-"*80)
    pwf = PurgedWalkForward(
        n_splits=5,
        embargo_td=pd.Timedelta(minutes=30),
        expanding_window=True
    )

    print(f"Número de splits: {pwf.get_n_splits(X)}")
    print()

    for fold, (train_idx, test_idx) in enumerate(pwf.split(X, y)):
        print(f"Fold {fold + 1}:")
        print(f"  Train: {len(train_idx):4d} samples")
        print(f"  Test:  {len(test_idx):4d} samples")
        print()

    print("✅ Todos os testes completos!")

    # Comparação: K-Fold normal vs Purged K-Fold
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO: K-FOLD NORMAL VS PURGED K-FOLD")
    print("="*80)

    from sklearn.model_selection import KFold

    # K-Fold normal
    kf_normal = KFold(n_splits=5, shuffle=False)
    # Purged K-Fold
    kf_purged = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=2))

    print("\nK-FOLD NORMAL (pode vazar informação):")
    total_train_normal = 0
    for i, (train_idx, test_idx) in enumerate(kf_normal.split(X)):
        total_train_normal += len(train_idx)
        print(f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")

    print(f"\nTotal train samples: {total_train_normal}")

    print("\nPURGED K-FOLD (protegido contra leakage):")
    total_train_purged = 0
    for i, (train_idx, test_idx) in enumerate(kf_purged.split(X)):
        total_train_purged += len(train_idx)
        print(f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")

    print(f"\nTotal train samples: {total_train_purged}")

    reduction = (total_train_normal - total_train_purged) / total_train_normal * 100
    print(f"\n⚠️  Redução de {reduction:.1f}% nos samples de treino devido ao purging")
    print("   Isso PREVINE overfitting por leakage temporal!")

    print("\n" + "="*80)
