# fx-candle-scanner

Forex scanner jo price action, candle reaction aur buyer/seller positions ko mila kar
agli candle ke liye BUY / SELL / HOLD batata hai, saath mein historical (out-of-sample) win%.

## Install
    pip install -r requirements.txt

## Run
    python fx_scanner.py                              # menu
    python fx_scanner.py --pairs EURUSD GBPJPY --detail
    python fx_scanner.py --all --interval 5m
    python fx_scanner.py --csv tv_export.csv --name EURJPY --interval 5m

## Note
- Win% prediction nahi, is rule ka purana hit-rate hai (train/test split ke saath).
- Binary payout par breakeven = 1 / (1 + payout). 85% payout par ~54.1%.
- Sirf real forex data (Yahoo / TradingView CSV). Broker OTC charts support nahi.
- Education/research ke liye. Financial advice nahi.
