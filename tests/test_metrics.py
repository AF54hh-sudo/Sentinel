import pandas as pd


def test_net_revenue_metric():
    sales = pd.DataFrame({"net_revenue": [10.25, 20.75, 30.0]})
    assert sales.net_revenue.sum() == 61.0

