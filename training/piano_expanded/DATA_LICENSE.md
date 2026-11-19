# 本輪資料來源與授權

本資料夾僅供本機非商業研究。不同來源的錄音維持各自授權，不能因其中包含 CC0 素材，就把整個資料集合或微調結果宣稱為無限制商用。

- **MAESTRO v3**：Google LLC、International Piano-e-Competition，CC BY-NC-SA 4.0。引用 Hawthorne et al., *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*, ICLR 2019。[官方頁面](https://magenta.withgoogle.com/datasets/maestro)、[授權](https://creativecommons.org/licenses/by-nc-sa/4.0/)。沿用前一輪本機四筆錄音。
- **Saarland Music Data v2**：Meinard Müller、Verena Konz、Vlora Arifi-Müller、Wolfgang Bogler、Johannes Zeitler。CC BY-NC-SA 3.0。引用 Müller et al., *Saarland Music Data*, ISMIR Late-Breaking and Demo Session, 2011。[官方說明](https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi)、[固定版本](https://zenodo.org/records/13753319)、[授權](https://creativecommons.org/licenses/by-nc-sa/3.0/)。只使用 wav_44100_stereo 原始錄音，未使用合成 MIDI 音訊。
- **Open Goldberg Variations**：J. S. Bach 作曲，Kimiko Ishizaka 演奏，CC0 1.0。使用 Aria 與 Variatio 13。[專案官方頁面](https://opengoldbergvariations.org/)、[音訊與授權中繼資料](https://archive.org/details/OpenGoldbergVariations)、[CC0](https://creativecommons.org/publicdomain/zero/1.0/)。

修改方式：擷取不重疊的 20 秒片段，首尾各加 20 毫秒淡入淡出，另存 PCM16 WAV。未聲稱原作者支持本模型。sources.json 保存下載網址、檔名、SHA256 與授權；clips.json 保存片段起點。原始檔保留不變。
