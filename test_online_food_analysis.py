import unittest
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class TestOnlineFoodAnalysis(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('onlinefoods.csv')
        self.df.drop(columns='Unnamed: 12', inplace=True)

    def test_data_load(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertGreater(self.df.shape[0], 0)
        self.assertGreater(self.df.shape[1], 0)

    def test_data_cleaning(self):
        self.assertNotIn('Unnamed: 12', self.df.columns)

    def test_age_distribution(self):
        age_counts = self.df['Age'].value_counts().sort_index()
        self.assertGreaterEqual(len(age_counts), 1)

    def test_chi_square(self):
        contingency_table = pd.crosstab(self.df['Marital Status'], self.df['Educational Qualifications'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        self.assertGreaterEqual(chi2, 0)
        self.assertGreaterEqual(p, 0)
        self.assertGreaterEqual(dof, 0)

    def test_income_conversion(self):
        self.df.loc[self.df['Monthly Income'] == '10001 to 25000', 'Monthly Income'] = 17500
        self.df.loc[self.df['Monthly Income'] == '25001 to 50000', 'Monthly Income'] = 37500
        self.df.loc[self.df['Monthly Income'] == 'Below Rs.10000', 'Monthly Income'] = 10000
        self.df.loc[self.df['Monthly Income'] == 'More than 50000', 'Monthly Income'] = 50000
        self.df.loc[self.df['Monthly Income'] == 'No Income', 'Monthly Income'] = 0
        self.df['Monthly Income'] = self.df['Monthly Income'].astype(int)
        self.assertTrue(self.df['Monthly Income'].dtype == int)


if __name__ == '__main__':
    unittest.main()
