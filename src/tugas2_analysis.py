import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    display = print

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
sns.set_theme(style='whitegrid', palette='deep')

file_path = 'Tugas1_DW.csv'
df = pd.read_csv(file_path)

display(df.head(10))
display(df.tail(10))
display(df.sample(10, random_state=42))

summary_soal1 = pd.DataFrame({
    'Metrik': ['Jumlah Record (Baris)', 'Jumlah Atribut (Kolom)', 'Daftar Nama Atribut'],
    'Nilai': [df.shape[0], df.shape[1], ', '.join(df.columns.tolist())]
})
display(summary_soal1)

dimensi_df = pd.DataFrame({'Jumlah Baris': [df.shape[0]], 'Jumlah Kolom': [df.shape[1]]})
display(dimensi_df)

struktur_df = pd.DataFrame({
    'Atribut': df.columns,
    'Tipe Data Awal (Pandas)': df.dtypes.astype(str).values,
    'Kelompok Atribut Awal': [
        'Teks / Identifier',
        'Teks / Identifier',
        'Teks / URL',
        'Teks (Ulasan)',
        'Numerik (Diskrit) / Kategorikal Ordinal',
        'Numerik (Diskrit)',
        'Kategorikal (Versi)',
        'Tanggal/Waktu (Tersimpan sebagai Object)',
        'Teks (Balasan Admin)',
        'Tanggal/Waktu (Tersimpan sebagai Object)',
        'Kategorikal (Versi)'
    ],
    'Kesesuaian Tipe Data': [
        'Sesuai',
        'Sesuai',
        'Sesuai',
        'Sesuai',
        'Sesuai',
        'Sesuai',
        'Sesuai',
        'Tidak Sesuai (Harus datetime64)',
        'Sesuai',
        'Tidak Sesuai (Harus datetime64)',
        'Sesuai'
    ]
})
display(struktur_df)

kelompok_ringkasan = pd.DataFrame({
    'Jenis Data': ['Numerik', 'Kategorikal', 'Teks / Identifier', 'Tanggal / Waktu'],
    'Daftar Atribut': [
        'score, thumbsUpCount',
        'reviewCreatedVersion, appVersion (serta score sebagai rating ordinal)',
        'reviewId, userName, userImage, content, replyContent',
        'at, repliedAt'
    ],
    'Jumlah Atribut': [2, 2, 5, 2]
})
display(kelompok_ringkasan)

missing_count = df.isnull().sum()
missing_pct = (missing_count / len(df)) * 100
unique_count = df.nunique(dropna=True)

kualitas_tabel = pd.DataFrame({
    'Atribut': df.columns,
    'Tipe Data': df.dtypes.astype(str).values,
    'Jumlah Non-Null': df.notnull().sum().values,
    'Jumlah Missing Values': missing_count.values,
    'Persentase Missing (%)': missing_pct.round(2).values,
    'Jumlah Nilai Unik': unique_count.values
}).sort_values(by='Jumlah Missing Values', ascending=False).reset_index(drop=True)
display(kualitas_tabel)

non_null_ver = df[['reviewCreatedVersion', 'appVersion']].dropna()
versi_identik = (non_null_ver['reviewCreatedVersion'] == non_null_ver['appVersion']).all()
reply_konsisten = (df['replyContent'].notnull() == df['repliedAt'].notnull()).all()

pemeriksaan_lain = pd.DataFrame({
    'Pemeriksaan Kualitas Data': [
        'Duplikat Seluruh Kolom (Full Row)',
        'Duplikat Berdasarkan reviewId',
        'Duplikat Berdasarkan userName',
        'Duplikat Berdasarkan Teks Ulasan (content)',
        'Konsistensi reviewCreatedVersion == appVersion (Non-Null)',
        'Konsistensi Pasangan replyContent & repliedAt',
        'Rentang Nilai score Valid (1 s/d 5)',
        'Nilai thumbsUpCount >= 0'
    ],
    'Hasil': [
        df.duplicated().sum(),
        df['reviewId'].duplicated().sum(),
        df['userName'].duplicated().sum(),
        df['content'].duplicated().sum(),
        versi_identik,
        reply_konsisten,
        df['score'].between(1, 5).all(),
        (df['thumbsUpCount'] >= 0).all()
    ]
})
display(pemeriksaan_lain)

df_clean = df.copy()
df_clean['at'] = pd.to_datetime(df_clean['at'], errors='coerce')
df_clean['repliedAt'] = pd.to_datetime(df_clean['repliedAt'], errors='coerce')

df_clean['char_count'] = df_clean['content'].astype(str).str.len()
df_clean['word_count'] = df_clean['content'].astype(str).str.split().str.len()
df_clean['reply_char_count'] = df_clean['replyContent'].fillna('').astype(str).str.len()
df_clean['response_time_hours'] = (df_clean['repliedAt'] - df_clean['at']).dt.total_seconds() / 3600.0

df_clean['has_reply'] = np.where(df_clean['replyContent'].notnull(), 'Dibalas', 'Tidak Dibalas')
df_clean['sentiment_category'] = pd.cut(
    df_clean['score'],
    bins=[0, 2, 3, 5],
    labels=['Negatif (1-2)', 'Netral (3)', 'Positif (4-5)']
)
df_clean['rating_label'] = 'Bintang ' + df_clean['score'].astype(str)
df_clean['appVersion_clean'] = df_clean['appVersion'].fillna('Unknown')

transparansi_data = pd.DataFrame({
    'Tahap Data': [
        'Data Awal Hasil Crawling',
        'Data Setelah Penghapusan Duplikat (reviewId)',
        'Data Valid untuk Analisis Utama (Tanpa Drop Baris Missing Opsional)',
        'Subset Data dengan Informasi Versi Aplikasi (appVersion Non-Null)',
        'Subset Data dengan Balasan Admin (replyContent Non-Null)'
    ],
    'Jumlah Record': [
        len(df),
        len(df_clean.drop_duplicates(subset=['reviewId'])),
        len(df_clean),
        df_clean['appVersion'].notnull().sum(),
        df_clean['replyContent'].notnull().sum()
    ],
    'Jumlah Atribut': [
        df.shape[1],
        df_clean.shape[1],
        df_clean.shape[1],
        df_clean.shape[1],
        df_clean.shape[1]
    ]
})
display(transparansi_data)
display(df_clean[['reviewId', 'score', 'thumbsUpCount', 'char_count', 'word_count', 'reply_char_count', 'has_reply', 'sentiment_category', 'appVersion_clean', 'at']].head(10))

num_cols = ['score', 'thumbsUpCount', 'char_count', 'word_count', 'reply_char_count']
stat_rows = []
for col in num_cols:
    s = df_clean[col].dropna()
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    stat_rows.append({
        'Atribut Numerik': col,
        'Count': len(s),
        'Mean': round(s.mean(), 4),
        'Median (Q2)': round(q2, 4),
        'Modus': s.mode().iloc[0] if not s.mode().empty else np.nan,
        'Std Dev': round(s.std(), 4),
        'Variance': round(s.var(), 4),
        'Minimum': s.min(),
        'Q1 (25%)': round(q1, 4),
        'Q3 (75%)': round(q3, 4),
        'Maksimum': s.max(),
        'IQR': round(q3 - q1, 4),
        'Skewness': round(s.skew(), 4)
    })
stat_numerik_df = pd.DataFrame(stat_rows)
display(stat_numerik_df)

cat_cols = ['rating_label', 'sentiment_category', 'has_reply', 'appVersion', 'reviewCreatedVersion']
cat_rows = []
for col in cat_cols:
    vc = df_clean[col].value_counts(dropna=True)
    total_valid = vc.sum()
    cat_rows.append({
        'Atribut Kategorikal': col,
        'Jumlah Kategori Unik': df_clean[col].nunique(dropna=True),
        'Kategori Tertinggi (Dominan)': vc.index[0],
        'Frekuensi Tertinggi': int(vc.iloc[0]),
        'Persentase Tertinggi (%)': round((vc.iloc[0] / total_valid) * 100, 2),
        'Kategori Terendah': vc.index[-1],
        'Frekuensi Terendah': int(vc.iloc[-1]),
        'Persentase Terendah (%)': round((vc.iloc[-1] / total_valid) * 100, 4)
    })
stat_kategorikal_df = pd.DataFrame(cat_rows)
display(stat_kategorikal_df)

for col in ['rating_label', 'sentiment_category', 'has_reply']:
    freq_df = df_clean[col].value_counts().reset_index()
    freq_df.columns = [col, 'Frekuensi']
    freq_df['Persentase (%)'] = ((freq_df['Frekuensi'] / len(df_clean)) * 100).round(2)
    display(freq_df)

top_ver_df = df_clean['appVersion'].value_counts(dropna=False).head(10).reset_index()
top_ver_df.columns = ['appVersion', 'Frekuensi']
top_ver_df['Persentase (%)'] = ((top_ver_df['Frekuensi'] / len(df_clean)) * 100).round(2)
display(top_ver_df)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
plot_cols = [
    ('score', 'Distribusi Skor Rating Ulasan (score)', 'Skor Rating (1 - 5)', '#2b5c8f', False),
    ('thumbsUpCount', 'Distribusi Jumlah Likes Ulasan (thumbsUpCount)', 'Jumlah Thumbs Up', '#d95f02', True),
    ('char_count', 'Distribusi Panjang Karakter Ulasan (char_count)', 'Jumlah Karakter', '#1b9e77', False),
    ('word_count', 'Distribusi Jumlah Kata Ulasan (word_count)', 'Jumlah Kata', '#7570b3', False)
]

for ax, (col, title, xlabel, color, use_log) in zip(axes.flatten(), plot_cols):
    data = df_clean[col].dropna()
    if col == 'score':
        sns.countplot(x=data, ax=ax, color=color, edgecolor='black')
    else:
        sns.histplot(data, bins=40, kde=not use_log, ax=ax, color=color, edgecolor='black')
        if use_log:
            ax.set_yscale('log')
    mean_val = data.mean()
    median_val = data.median()
    skew_val = data.skew()
    if col != 'score':
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.8, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='gold', linestyle='-', linewidth=1.8, label=f'Median: {median_val:.2f}')
        ax.legend()
    ax.set_title(f'{title}\n(Skewness: {skew_val:.2f})', fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('Frekuensi (Log Scale)' if use_log else 'Frekuensi', fontsize=10)

plt.tight_layout()
plt.savefig('soal5_histogram_numerik.png', dpi=300, bbox_inches='tight')
plt.show()

outlier_rows = []
for col in ['score', 'thumbsUpCount', 'char_count', 'word_count']:
    s = df_clean[col].dropna()
    q1 = s.quantile(0.25)
    median = s.quantile(0.50)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    outlier_rows.append({
        'Atribut': col,
        'Q1 (25%)': round(q1, 2),
        'Median (50%)': round(median, 2),
        'Q3 (75%)': round(q3, 2),
        'IQR': round(iqr, 2),
        'Batas Bawah (Q1 - 1.5*IQR)': round(lower, 2),
        'Batas Atas (Q3 + 1.5*IQR)': round(upper, 2),
        'Jumlah Outlier': len(outliers),
        'Persentase Outlier (%)': round((len(outliers) / len(s)) * 100, 2)
    })
outlier_df = pd.DataFrame(outlier_rows)
display(outlier_df)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
box_specs = [
    ('score', 'Boxplot Skor Rating Ulasan (score)', 'Skor Rating', '#2b5c8f'),
    ('thumbsUpCount', 'Boxplot Jumlah Thumbs Up (thumbsUpCount)', 'Jumlah Thumbs Up', '#d95f02'),
    ('char_count', 'Boxplot Panjang Karakter Ulasan (char_count)', 'Jumlah Karakter', '#1b9e77'),
    ('word_count', 'Boxplot Jumlah Kata Ulasan (word_count)', 'Jumlah Kata', '#7570b3')
]

for ax, (col, title, xlabel, color) in zip(axes.flatten(), box_specs):
    sns.boxplot(x=df_clean[col], ax=ax, color=color, fliersize=4, width=0.45)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig('soal6_boxplot_outlier.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

rating_order = ['Bintang 1', 'Bintang 2', 'Bintang 3', 'Bintang 4', 'Bintang 5']
rating_counts = df_clean['rating_label'].value_counts().reindex(rating_order)
bars1 = axes[0, 0].bar(rating_counts.index, rating_counts.values, color=['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#1a9850'], edgecolor='black')
axes[0, 0].set_title('Bar Chart Distribusi Rating Bintang Ulasan', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Kategori Rating', fontsize=10)
axes[0, 0].set_ylabel('Jumlah Ulasan', fontsize=10)
for bar in bars1:
    h = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, h + 60, f'{int(h)} ({h/len(df_clean)*100:.1f}%)', ha='center', fontsize=9)
axes[0, 0].set_ylim(0, max(rating_counts.values) * 1.15)

top10_ver = df_clean['appVersion'].dropna().value_counts().head(10)
bars2 = axes[0, 1].barh(top10_ver.index[::-1], top10_ver.values[::-1], color='#386cb0', edgecolor='black')
axes[0, 1].set_title('Bar Chart 10 Versi Aplikasi Teratas (appVersion)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Jumlah Ulasan', fontsize=10)
axes[0, 1].set_ylabel('Versi Aplikasi', fontsize=10)
for bar in bars2:
    w = bar.get_width()
    axes[0, 1].text(w + 15, bar.get_y() + bar.get_height()/2, f'{int(w)} ({w/df_clean["appVersion"].notnull().sum()*100:.1f}%)', va='center', fontsize=9)
axes[0, 1].set_xlim(0, max(top10_ver.values) * 1.2)

sent_counts = df_clean['sentiment_category'].value_counts()
axes[1, 0].pie(
    sent_counts.values,
    labels=sent_counts.index,
    autopct='%1.2f%%',
    startangle=140,
    colors=['#1a9850', '#d73027', '#fee08b'],
    explode=(0.04, 0.04, 0.04),
    wedgeprops={'edgecolor': 'black'}
)
axes[1, 0].set_title('Pie Chart Proporsi Sentimen Ulasan Berdasarkan Rating', fontsize=12, fontweight='bold')

reply_counts = df_clean['has_reply'].value_counts()
axes[1, 1].pie(
    reply_counts.values,
    labels=reply_counts.index,
    autopct='%1.2f%%',
    startangle=90,
    colors=['#66c2a5', '#fc8d62'],
    explode=(0.03, 0.03),
    wedgeprops={'edgecolor': 'black'}
)
axes[1, 1].set_title('Pie Chart Proporsi Status Balasan Admin (has_reply)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('soal7_bar_pie_kategorikal.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

sns.scatterplot(
    data=df_clean, x='word_count', y='char_count',
    hue='sentiment_category', alpha=0.6, ax=axes[0]
)
axes[0].set_title('Scatter Plot: Jumlah Kata vs Panjang Karakter', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Jumlah Kata (word_count)', fontsize=10)
axes[0].set_ylabel('Jumlah Karakter (char_count)', fontsize=10)

sns.scatterplot(
    data=df_clean, x='word_count', y='thumbsUpCount',
    hue='sentiment_category', alpha=0.6, ax=axes[1]
)
axes[1].set_title('Scatter Plot: Jumlah Kata vs Thumbs Up Count', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Jumlah Kata (word_count)', fontsize=10)
axes[1].set_ylabel('Jumlah Thumbs Up (thumbsUpCount)', fontsize=10)

sns.scatterplot(
    data=df_clean, x='char_count', y='score',
    hue='has_reply', alpha=0.6, ax=axes[2]
)
axes[2].set_title('Scatter Plot: Panjang Karakter vs Skor Rating', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Jumlah Karakter (char_count)', fontsize=10)
axes[2].set_ylabel('Skor Rating (score)', fontsize=10)

plt.tight_layout()
plt.savefig('soal8_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

corr_cols = ['score', 'thumbsUpCount', 'char_count', 'word_count', 'reply_char_count']
corr_matrix = df_clean[corr_cols].corr(method='pearson')
display(corr_matrix.round(4))

plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.4f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    linewidths=0.8,
    cbar_kws={'label': 'Koefisien Korelasi Pearson (r)'}
)
plt.title('Correlation Heatmap Atribut Numerik (Korelasi Pearson)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('soal9_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        c1, c2 = corr_cols[i], corr_cols[j]
        r_val = corr_matrix.loc[c1, c2]
        pairs.append({
            'Pasangan Atribut': f'{c1} vs {c2}',
            'Korelasi Pearson (r)': round(r_val, 4),
            'Korelasi Absolut (|r|)': round(abs(r_val), 4),
            'Arah Hubungan': 'Positif' if r_val > 0 else 'Negatif'
        })
pairs_df = pd.DataFrame(pairs).sort_values(by='Korelasi Absolut (|r|)', ascending=False).reset_index(drop=True)
display(pairs_df)
display(pairs_df.head(3))
