# Guitar IR Analyzer V2.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ir-analyzer.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Guitar IR Analyzer** は、ギターキャビネットのインパルス応答（IR: Impulse Response）ファイルをアップロードし、時間領域（波形）および周波数領域（FFT）で比較・解析するためのWebアプリケーションです。

複数のIRファイルを同時に比較し、詳細な信号処理設定（ウィンドウ関数、スムージング等）を適用した上で、論文やドキュメント作成に耐えうる高解像度なグラフ画像を出力することが可能です。

🔗 **Live Demo:** [https://ir-analyzer.streamlit.app/](https://ir-analyzer.streamlit.app/)

## 📖 目次

- [特徴 (Features)](#-特徴-features)
- [技術的詳細 (Technical Details)](#-技術的詳細-technical-details)
- [インストールと実行 (Installation)](#-インストールと実行-installation)
- [使い方 (Usage)](#-使い方-usage)
- [使用ライブラリ (Requirements)](#-使用ライブラリ-requirements)
- [Author](#-author)

## ✨ 特徴 (Features)

このアプリケーションは、音響機器の自作や解析を行うエンジニア・ミュージシャンのための強力な機能を備えています。

### 1. 高度な波形解析
* **マルチファイル対応**: 最大8つまでのWAVファイルを同時にアップロードし、重ね合わせて比較可能。
* **ノーマライズ処理**: アップロードされたIRデータは自動的に正規化され、振幅比較が容易になります。
* **可変表示範囲**: 1ms〜100msの範囲で、初期反射やトランジェントを詳細に確認できます。

### 2. 詳細な周波数解析 (FFT)
* **ウィンドウ関数の選択**: 解析目的に応じて、以下の9種類のウィンドウ関数を選択可能です。
    * Rectangular (Boxcar), Hann, Hamming, Blackman, Bartlett, Kaiser, Blackman-Harris, Flat Top, Tukey
* **オクターブ・スムージング**: 聴感上の特性に近い表示を行うため、1/48オクターブから1オクターブまでのスムージング処理を適用可能。
* **FFT設定の最適化**: ゼロ埋め（Zero-padding）による周波数分解能の向上や、FFTサイズ（2048〜65536）の手動指定に対応。

### 3. 高品質なエクスポート機能
* **高解像度出力**: 最大 **2000 DPI** までの高解像度PNG画像としてグラフをダウンロード可能。研究資料や技術ブログへの掲載に最適です。
* **視認性のカスタマイズ**: ダークモード/ライトモード対応に加え、グリッド線の濃さ、テキストサイズ、配色パターン（Alphaブレンド、グラデーション等）を調整できます。

## 🔧 技術的詳細 (Technical Details)

本ツールは `scipy.signal` および `numpy.fft` を使用して信号処理を行っています。

* **FFT処理**: 指定された時間長（Duration）に基づいて必要なサンプル数を計算し、不足分はゼロパディングを行うことで、短いIRファイルでも十分な周波数分解能を確保しています。
* **振幅計算**: パワースペクトルではなく振幅スペクトルを計算し、最大値を基準としたdB（デシベル）正規化を行って表示しています。
* **平滑化アルゴリズム**: 指定されたオクターブ幅（例: 1/24 Oct）に基づいて移動平均フィルタを適用し、周波数レスポンスの概形を視覚化しやすくしています。

## 🚀 インストールと実行 (Installation)

ローカル環境で実行する場合は、以下の手順に従ってください。

### 前提条件
* Python 3.9 以上推奨

### セットアップ手順

1. **リポジトリのクローン**
   ```bash
   git clone [https://github.com/YourUsername/IR-Analyzer.git](https://github.com/YourUsername/IR-Analyzer.git)
   cd IR-Analyzer
   ```

2. **仮想環境の作成 (推奨)**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```

4. **アプリケーションの起動**
   ```bash
   streamlit run app.py
   ```
   ブラウザが自動的に開き、アプリが表示されます。

## 📝 使い方 (Usage)

1.  **File Upload**: 左側のサイドバー、またはメイン画面のドラッグ＆ドロップエリアにWAV形式のIRファイルをアップロードします（最大8ファイル）。
2.  **Waveform Settings**: サイドバーの "Waveform Display Duration" で時間軸の表示範囲を調整します。
3.  **FFT Settings**:
    * **Duration**: FFT解析に使用するサンプル長を指定します。
    * **Window Function**: 解析用途に合わせてウィンドウ関数を選択します（通常は `Hann` や `Blackman-Harris` を推奨）。
    * **Smoothing**: グラフがギザギザして見にくい場合は、`1/24 Octave` 程度にスムージングを設定してください。
4.  **Export**: グラフ下部の "Download High-Resolution..." ボタンから画像を保存します。DPI設定はサイドバー下部で行えます。

## 📦 使用ライブラリ (Requirements)

* [Streamlit](https://streamlit.io/) - Webアプリケーションフレームワーク
* [NumPy](https://numpy.org/) - 数値計算
* [Matplotlib](https://matplotlib.org/) - グラフ描画
* [SciPy](https://scipy.org/) - 信号処理
