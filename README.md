# Doubutsushougi-solver

どうぶつしょうぎ（3x4盤）の局面を探索し、手番側の勝敗評価と最善手候補を返す最小実装です。

## ルール仕様（本実装）

- 盤面: 3列 x 4行
- 駒: ライオン(L)、キリン(G)、ゾウ(E)、ひよこ(C)、にわとり(H)
- 手番: `b`（先手/Black）、`w`（後手/White）
- 成り: ひよこが相手最奥段に移動したとき自動でにわとりに成る
- 持ち駒: ひよこ/キリン/ゾウのみ（ライオン捕獲で即終局、にわとり捕獲時はひよことして持ち駒化）
- 勝敗条件:
  - 相手ライオンを取る
  - トライ（自ライオンが相手最奥段に到達し、相手に即時捕獲されない）

## 実装構成

- `src/doubutsu/game.py`: 局面表現、合法手生成、着手適用、終局判定
- `src/doubutsu/solver.py`: 深さ制限付きNegamax探索
- `src/doubutsu/cli.py`: CLI
- `main.py`: エントリポイント
- `tests/`: 単体テスト

## 入力形式（SFEN風）

CLIの `--position` には `initial` または以下形式を指定できます。

`<board> <turn> <black_hand> <white_hand>`

- `board`: 4段を `/` 区切り、各段3文字
  - `.`: 空きマス
  - `L,G,E,C,H`: 先手駒
  - `l,g,e,c,h`: 後手駒
- `turn`: `b` or `w`
- `black_hand`, `white_hand`: 持ち駒文字列（`C/G/E` の並び、なしは `-`）

例（初期局面相当）:

`gle/.c./.C./ELG b - -`

## 実行方法

```bash
python main.py --position initial --depth 8
```

任意局面:

```bash
python main.py --position ".l./.C./.../.L. b - -" --depth 4
```

## テスト

```bash
python -m unittest discover -s tests -v
```

## 制約

- 探索は深さ制限付きで、デフォルトでは完全解析ではありません。
- 千日手や詳細な反復局面規則は未実装です。
