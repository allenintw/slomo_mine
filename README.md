# slomo_mine

# 巢狀模型用於醫學影片幀插值


影片幀插值 (Video Frame Interpolation)的目的主要在於從給定的兩個連續影
像幀中，透過判斷影像中物體移動的方向與速率來得到光流 (Optical Flow)，藉此
生成 物體 移動過程中 的 中間幀 。 而目前的 影片 幀插值多 著重 於單 一 中間幀的生成
但 隨著 需 求的 增加 ，多 個中間 幀 的 影片幀 插值 技術 也 逐漸 受到 重視 因此我們 提
出 一個可 用於多幀插值 的 巢狀架構 卷積網路 模型 。

為了 使 模型有多幀插值的能力 ，我們 使用 兩階段 的 光流計算架構 來得 到任意
時間點的光流 在 第一階段的 子 架構 中，我們首先 採用 U-net架構 進行 光流計算
接著利用 線性計算 來得 到任意時間 點的光流 之後在 第二階段的子架構中 採用 巢
狀的 U2-net模型 來針對 任意時間點的光流進行 優化 。 然 而 ，這樣的模型架構 在
處理 大位移 運動 或是多幀插值 的 光流計算能力 還是 略顯不足 ，因此 ，我們使用 側
邊輸出的方式來強化 模型 在多尺度與多 階層的 特徵傳遞 能 力 讓模型 在針對 大位
移的運動情況 時 ，能夠 有更 好的 預測效果 。

與此 同時 ，我們 也將影片幀插值的技術 用於 特殊的 X光 成像 的 醫學影片中
並且針對 大 位移的運 動 進行 單幀插值 、小位移的運動 進行 多幀插值 ，從 實驗的 結
果表明 ，我們的方法 在單幀插值的 結果 比現有的 方法 來得 更好 ，而在多幀插值的
任務 上 我們的 方法 雖然 在 相對 低 幀 的 生成 結果 略 差於 其他方法 ，然而隨著 預測
中間幀的數量增加 ，我們的方法相較於 其他方法更具有優勢 。

關鍵詞：影片幀插值、
多幀插值 、巢狀結構 、 X-ray醫學影片 。

# 環境建立

Anaconda虛擬環境
```
conda create -n slomo_test
conda activate slomo_test
```

套件安裝
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install scikit-image
pip install tqdm
pip install natsort
pip install ffmpeg
conda install -c conda-forge tensorboardx
conda install -c conda-forge nvidia-apex
```
