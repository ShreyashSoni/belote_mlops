# -*- coding: utf-8 -*-
"""
Unit Testing suite for model/model_features
Run the tests:
Terminal    => pytest --cov=./ tests/ -v
HTML Report => pytest --cov-report html:/belote/htmlcov --cov=./ tests/ -v
"""
import pytest
import pandas as pd

from model.model_features import CustomFeaturesBuilder


@pytest.fixture
def custom_feature_builder():  # pylint: disable=redefined-outer-name, missing-function-docstring
    return CustomFeaturesBuilder


@pytest.fixture
def synthetic_df() -> pd.DataFrame:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return pd.DataFrame(
        {
            "reward": [165, -165, 1, 2],
            "last_bidder": [1, 2, 1, 2],
            "starter": [2, 1, 2, 1],
            "contract": ["hearts", "spades", "diamonds", "clubs"],
            "p1_face_value": [10, 20, 30, 40],
            "p2_face_value": [20, 30, 40, 50],
        }
    )


@pytest.fixture
def features_df() -> pd.DataFrame:  # pylint: disable=redefined-outer-name, missing-function-docstring
    return pd.DataFrame(
        {
            "has_BR_at_clubs": [1, 2, 3],
            "has_BR_at_diamonds": [1, 2, 3],
            "has_BR_at_hearts": [1, 2, 3],
            "has_BR_at_spades": [1, 2, 3],
            "has_tierce_at_clubs": [2, 3, 4],
            "has_tierce_at_diamonds": [2, 3, 4],
            "has_tierce_at_hearts": [2, 3, 4],
            "has_tierce_at_spades": [2, 3, 4],
            "total_BR_points": [4, 8, 12],
            "total_tierce_points": [8, 12, 16],
            "total_AnD_points": [12, 20, 28],
        }
    )


def test_custom_feature_total_br_points_works_correctly(  # pylint: disable=redefined-outer-name
    mocker, custom_feature_builder, features_df
):
    """ We create the relevant features based on Belote Rebelote points """
    intermediate_df = features_df.drop(
        ["total_BR_points", "total_tierce_points", "total_AnD_points"], axis=1
    )
    features_df = features_df.drop(["total_tierce_points", "total_AnD_points"], axis=1)
    processed_output: pd.DataFrame = (
        custom_feature_builder(mocker.MagicMock(), intermediate_df)
        .feature_total_br_points()
        .features_df
    )
    pd.testing.assert_frame_equal(processed_output, features_df)


def test_custom_feature_total_tierce_points_works_correctly():
    """NA"""


def test_custom_feature_total_and_points_works_correctly():
    """NA"""


def test_balanced_resampling_works_correctly(
        custom_feature_builder, synthetic_df, mocker
):
    """We resample the data by getting rid of 50% of the unbalanced set"""
    processed_output: pd.DataFrame = (
        custom_feature_builder(synthetic_df, mocker.MagicMock())
        .balanced_resampling(sampling=0.5)
        .synthetic_df
    )
    assert processed_output.shape == (3, 6)


def test_custom_feature_merge_synthetic_and_features_works_correctly():
    """NA"""


def test_custom_feature_encode_contract_works_correctly():
    """NA"""


def test_custom_feature_categorize_reward_works_correctly():
    """NA"""
