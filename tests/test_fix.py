import sys
from unittest.mock import MagicMock

# Mock necessary modules that main.py imports but we don't need for testing sanitization
tf = MagicMock()
tf.data.AUTOTUNE = -1
sys.modules['tensorflow'] = tf
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.layers'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['data_augmentation'] = MagicMock()
sys.modules['model_components'] = MagicMock()
sys.modules['data_preprocessing'] = MagicMock()

# Mock pandas as well since it's not installed
sys.modules['pandas'] = MagicMock()

# Now we can import from main
try:
    from main import _sanitize_val, _sanitize_for_excel
except ImportError as e:
    print(f"Failed to import from main: {e}")
    sys.exit(1)

class MockSeries:
    def __init__(self, data, dtype):
        self.data = data
        self.dtype = dtype
    def apply(self, func):
        return [func(x) for x in self.data]

class MockDataFrame:
    def __init__(self, df_dict, dtypes):
        self.df_dict = df_dict
        self.dtypes_dict = dtypes
    def copy(self):
        return MockDataFrame(self.df_dict.copy(), self.dtypes_dict.copy())
    @property
    def columns(self):
        return list(self.df_dict.keys())
    def __getitem__(self, col):
        class ColumnProxy:
            def __init__(self, data, dtype):
                self.data = data
                self.dtype = dtype
            def apply(self, func):
                return [func(x) for x in self.data]
        return ColumnProxy(self.df_dict[col], self.dtypes_dict[col])
    def __setitem__(self, col, val):
        self.df_dict[col] = val

def test_fix():
    # Test _sanitize_val with all risky characters
    print("Testing _sanitize_val...")
    assert _sanitize_val("=SUM(1,2)") == "'=SUM(1,2)"
    assert _sanitize_val("+1+2") == "'+1+2"
    assert _sanitize_val("-1-2") == "'-1-2"
    assert _sanitize_val("@at") == "'@at"
    assert _sanitize_val("\ttab") == "'\ttab"
    assert _sanitize_val("\rreturn") == "'\rreturn"
    assert _sanitize_val("safe") == "safe"
    assert _sanitize_val(123) == 123
    print("  _sanitize_val tests passed.")

    # Test _sanitize_for_excel with MockDataFrame
    print("Testing _sanitize_for_excel...")
    data = {
        "filename": ["=SUM(1,2).png", "normal.png", "+add.jpg"],
        "value": [1.0, 2.0, 3.0],
        "misc": ["@at", "\ttab", "safe"]
    }
    dtypes = {
        "filename": "object",
        "value": "float64",
        "misc": "object"
    }

    df = MockDataFrame(data, dtypes)
    sanitized_df = _sanitize_for_excel(df)

    expected_filename = ["'=SUM(1,2).png", "normal.png", "'+add.jpg"]
    expected_misc = ["'@at", "'\ttab", "safe"]

    assert sanitized_df.df_dict["filename"] == expected_filename
    assert sanitized_df.df_dict["misc"] == expected_misc
    assert sanitized_df.df_dict["value"] == [1.0, 2.0, 3.0]
    print("  _sanitize_for_excel tests passed.")

    print("\nVerification successful! All malicious patterns were correctly prefixed using production logic.")

if __name__ == "__main__":
    test_fix()
