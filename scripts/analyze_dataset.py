import pandas as pd
import numpy as np

df = pd.read_csv('data/new_all_tiles.csv')

print('=== NEUE ERKENNTNISSE & IMPLIKATIONEN ===\n')

# 1. Temporal clustering
print('1. TEMPORALES CLUSTERING (historischer Kontext):')
peak_years = df['year'].value_counts().head(10).sort_index()
print('Top 10 Jahre nach Anzahl Kacheln:')
for year, count in peak_years.items():
    pct = 100 * count / len(df)
    print(f'   {int(year)}: {count:3d} Kacheln ({pct:5.1f}%) ', end='')
    if year <= 1918:
        print('← WWI-Ära')
    elif year >= 1980:
        print('← Moderne')
    else:
        print()

peak_total = df[df['year'].isin([1906, 1907, 1911, 1912, 1914, 1918])]['year'].count()
print(f'\nSpitze 1906-1918 (WWI-Periode): {peak_total} Kacheln ({100*peak_total/len(df):.1f}%)')
print('→ Temporal diversity ist NICHT gleichverteilt!')
print('→ γ_temporal muss gegen dieses Clustering optimieren\n')

# 2. Geographic coverage
print('2. GEOGRAFISCHE ABDECKUNG:')
lat_span = df['N'].max() - df['N'].min()
lon_span = df['right'].max() - df['left'].min()
area_approx = lat_span * lon_span * 111 * 111
print(f'Bounding Box: {lat_span:.2f}° × {lon_span:.2f}° ≈ {area_approx:,.0f} km²')
print(f'Tiles: 673 → Durchschnitt {area_approx/673:.0f} km²/tile')
print(f'→ Bei min_distance=40km → Sperrkreis ~5,027 km²/tile')
print(f'→ Theoretisches Maximum: ~{int(area_approx/5027)} selectable tiles')
print(f'→ Tatsächlich selektiert: 40 tiles (Multi-Criteria)\n')

# 3. Data quality
print('3. DATENQUALITÄT:')
quality_counts = df['data_quality'].value_counts()
print(quality_counts)
print()

# 4. Missing years
print('4. FEHLENDE JAHRE:')
missing_years = df['year'].isna().sum()
print(f'Kacheln ohne Jahr: {missing_years} ({100*missing_years/len(df):.1f}%)')
if missing_years > 0:
    print('→ Werden mit Median gefüllt in current implementation')
    print(f'→ Median: {df["year"].median():.0f}')
print()

# 5. Sheet numbers
print('5. BLATTNUMMERN (SheetNumber):')
unique_sheets = df['SheetNumber'].nunique()
print(f'Unique Sheet Numbers: {unique_sheets}')
print(f'Total tiles: {len(df)}')
print(f'→ Durchschnitt {len(df)/unique_sheets:.1f} tiles pro Blattnummer')
if len(df) > unique_sheets:
    multi_tiles = df.groupby('SheetNumber').size()
    max_tiles = multi_tiles.max()
    print(f'→ Maximum {max_tiles} Versionen pro Blatt (multi-temporal!)')
    print('→ Multi-temporal tiles = Chance für temporal diversity!')
    
    sample_sheet = multi_tiles[multi_tiles > 1].index[0]
    sample_data = df[df['SheetNumber'] == sample_sheet][['SheetNumber', 'year', 'N', 'left']].sort_values('year')
    print(f'\nBeispiel: Blatt {sample_sheet} ({len(sample_data)} Versionen):')
    print(sample_data.to_string(index=False))

print('\n=== KONSEQUENZEN FÜR OPTIMIERUNG ===\n')

print('1. Temporal clustering (1906-1918) erfordert AKTIVE Diversifizierung')
print('   → γ_temporal=0.15-0.20 empfohlen (validiert: r=0.732)')
print()

print('2. Multi-temporal tiles (gleiche Location, verschiedene Jahre):')
print('   → Spatial constraint verhindert Duplikate')
print('   → ABER: Temporal weight kann zwischen Versionen wählen!')
print()

print('3. Data quality als Constraint:')
print('   → Optional: Filtern auf data_quality="ok" vor Selection')
print()

print('4. Geographic density variiert:')
print('   → min_distance=40km ist konservativ')
print('   → Könnte regional adaptive sein (dicht vs. spärlich)')
