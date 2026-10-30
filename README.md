# Detekcja punktów z wykresów

Projekt generuje syntetyczne wykresy punktowe, wykrywa znajdujące się na nich
punkty i porównuje wynik detekcji z metadanymi referencyjnymi (ground truth).
Zbiór obejmuje dwa style wykresów:

- `scatter_plot` — punkty bez linii łączącej,
- `line_scatter_plot` — punkty połączone czarną linią.

Głównym celem jest sprawdzenie, jak obecność linii oraz właściwości wykresu
wpływają na skuteczność wykrywania punktów.

> Projekt jest obecnie przygotowany do uruchamiania na systemie Windows.
> Konfiguracja lokalnego środowiska wirtualnego i instrukcje poniżej dotyczą
> wyłącznie Windows.

## Wymagania

- Windows 10 lub nowszy,
- Python 3.10+,
- PowerShell,
- VS Code z rozszerzeniem Python oraz Jupyter, jeśli notebook ma być
  uruchamiany interaktywnie.

Zależności projektu są zapisane w [requirements.txt](requirements.txt).

## Instalacja środowiska na Windows

W PowerShell przejdź do katalogu głównego repozytorium i utwórz środowisko:

```powershell
py -m venv cv_rek
```

Jeśli PowerShell blokuje aktywację skryptów, zezwól na nią tylko dla bieżącego
procesu:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Aktywuj środowisko i zainstaluj zależności:

```powershell
.\cv_rek\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Po aktywacji środowiska prefiks `(cv_rek)` powinien być widoczny w promptcie
PowerShella.

## Uruchomienie end-to-end

Wszystkie poniższe polecenia należy wykonywać z katalogu głównego repozytorium,
przy aktywnym środowisku `cv_rek`.

### 1. Generowanie danych

```powershell
python src\scatter_detection\generation\data_generation.py
```

Generator:

1. tworzy funkcje benchmarkowe,
2. losuje punkty zgodnie z pięcioma strategiami próbkowania osi X,
3. dodaje szum,
4. tworzy oba style wykresów,
5. zachowuje równowagę kombinacji cech przez stałą `N`,
6. zapisuje obraz PNG oraz jego metadane JSON.

Dane trafiają do:

```text
src\scatter_detection\data\generated_data\
├── images\
└── metadata\
```

Podczas generowania wyświetlany jest pasek postępu.

### 2. Detekcja punktów

```powershell
python src\scatter_detection\pipeline\main.py
```

Pipeline dla każdego obrazu:

1. odczytuje metadane i wyznacza geometrię obszaru wykresu,
2. izoluje obszar wykresu,
3. uruchamia metodę bazową,
4. uruchamia metodę `line_aware`,
5. przelicza współrzędne pikselowe na wartości osi,
6. zapisuje wyniki obu metod w jednym pliku JSON.

Dodatkowo pipeline zapisuje konfiguracje użyte do kalibracji w:

```text
src\scatter_detection\data\generated_data\configs\
```

Wyniki detekcji są zapisywane w:

```text
src\scatter_detection\data\generated_data\detected_points\
```

### 3. Przygotowanie tabeli ewaluacyjnej

```powershell
python src\scatter_detection\evaluation\data_preparation.py
```

Skrypt dopasowuje punkty przewidziane przez detektor do punktów ground truth
z użyciem tolerancji zależnej od zakresu osi. Następnie wylicza między innymi:

- precision,
- recall,
- F1,
- średnią odległość dopasowanych punktów,
- RMSE odległości.

Wyniki są zapisywane jako:

```text
src\scatter_detection\data\evaluation\
├── detection_evaluation.csv
└── detection_evaluation.parquet
```

### 4. Analiza notebookowa

Notebook znajduje się w:

[src/scatter_detection/evaluation/evaluation.ipynb](src/scatter_detection/evaluation/evaluation.ipynb)

Można otworzyć go w VS Code i wybrać interpreter z `cv_rek`, a następnie
uruchomić komórki po kolei. Notebook agreguje wyniki według:

- strategii próbkowania osi X,
- rozmiaru markera,
- stylu wykresu,
- typu funkcji,
- obecności siatki.

## Struktura repozytorium

```text
.
├── cv_rek\                         # lokalne środowisko wirtualne Windows
├── experiments\                    # eksperymenty i materiały pomocnicze
├── instructions\                  # dodatkowe instrukcje projektu
├── src\
│   ├── scatter_detection\
│   │   ├── config\
│   │   │   └── constants.py        # zakresy osi i parametry generatora
│   │   ├── data\
│   │   │   ├── generated_data\     # obrazy, metadane i wyniki detekcji
│   │   │   └── evaluation\        # CSV/Parquet z wynikami
│   │   ├── detection\
│   │   │   ├── configuration.py    # geometria obrazu i kalibracja
│   │   │   ├── point_detection.py  # maskowanie i detekcja punktów
│   │   │   └── readme.md
│   │   ├── evaluation\
│   │   │   ├── data_preparation.py # metryki i tabela wyników
│   │   │   └── evaluation.ipynb    # analiza i wykresy
│   │   ├── generation\
│   │   │   ├── data_generation.py  # generator obrazów i ground truth
│   │   │   └── helpers.py          # próbkowanie i funkcje benchmarkowe
│   │   └── pipeline\
│   │       └── main.py             # uruchomienie detekcji dla całego zbioru
│   └── __init__.py
├── requirements.txt
└── README.md
```

Podział kodu odzwierciedla przepływ danych: `generation` tworzy dane,
`detection` zawiera algorytmy, `pipeline` spina przetwarzanie, a `evaluation`
porównuje wyniki z ground truth. Katalog `data` przechowuje artefakty, a nie
logikę aplikacji.

## Jak działa detekcja

### Metoda bazowa

Metoda bazowa tworzy maskę ciemnych elementów obrazu, a następnie analizuje
spójne komponenty. Odrzuca komponenty o niepasującym polu powierzchni lub
proporcjach, aby ograniczyć wpływ osi i elementów tła.

### Metoda `line_aware`

Na wykresach `line_scatter_plot` linie łączące mogą zostać potraktowane jako
duże lub wydłużone obiekty. Metoda `line_aware` korzysta z transformacji
odległości i zachowuje zwarte rdzenie markerów, ograniczając wpływ cienkich
linii na wynik.

## Fundamentalne wyniki

Poniższe wartości pochodzą z aktualnego pliku
`src\scatter_detection\data\evaluation\detection_evaluation.csv`.
Zbiór zawiera 576 obrazów, czyli 1152 wiersze wyników — po jednym wyniku dla
każdej metody detekcji na każdym obrazie.

### Średnia metryk liczona po obrazach

| Metoda | Precision | Recall | F1 | Średnia odległość |
|---|---:|---:|---:|---:|
| `baseline` | 0.535 | 0.437 | 0.431 | 0.0333 |
| `line_aware` | 0.723 | 0.674 | 0.675 | 0.0207 |

Metoda `line_aware` poprawia średnie F1 o około 0.244 względem metody bazowej
i zmniejsza średnią odległość dopasowania o około 38%.

### Agregacja globalna po wszystkich punktach

| Metoda | Precision | Recall | F1 |
|---|---:|---:|---:|
| `baseline` | 0.618 | 0.427 | 0.505 |
| `line_aware` | 0.658 | 0.644 | 0.651 |

W ujęciu globalnym metoda `line_aware` szczególnie poprawia recall: z 0.427 do
0.644. Oznacza to, że skuteczniej odzyskuje punkty, które metoda bazowa traci,
zwłaszcza gdy obraz zawiera linie łączące.

### Wpływ stylu wykresu

Średnie F1 po obrazach dla obu stylów:

| Metoda | `scatter_plot` | `line_scatter_plot` |
|---|---:|---:|
| `baseline` | 0.795 | 0.067 |
| `line_aware` | 0.917 | 0.434 |

Najważniejszy wniosek jest zgodny z założeniem projektu: metoda bazowa działa
bardzo dobrze na samych punktach, ale obecność linii dramatycznie pogarsza jej
wynik. Wariant `line_aware` jest wyraźnie odporniejszy na ten przypadek,
chociaż wykresy punktowo-liniowe nadal są trudniejsze niż zwykłe wykresy
punktowe.

## Uwagi dotyczące wyników

Wyniki opisują konkretny, deterministyczny zbiór wygenerowany z użyciem
`RANDOM_SEED`. Ponowne wygenerowanie danych może nadpisać obrazy, metadane
i wyniki ewaluacji. Jeżeli zmienione zostaną parametry generatora, wartości
w sekcji wynikowej README należy traktować jako punkt odniesienia dla obecnego
artefaktu, a nie jako gwarant dla nowego zbioru.

Ponadto wygenerowane dane (zdjęcia, ich metadane i pliki konfiguracyjne) są ignorowane przez środowisko Git, tak aby nie zaśmiecać
repozytorium ogromną liczbą plików.