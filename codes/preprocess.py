import os
import glob
import shutil
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import *

num_cols = [
'age_approx',                        # Approximate age of patient at time of imaging.
'clin_size_long_diam_mm',            # Maximum diameter of the lesion (mm).+
'tbp_lv_A',                          # A inside  lesion.+
'tbp_lv_Aext',                       # A outside lesion.+
'tbp_lv_B',                          # B inside  lesion.+
'tbp_lv_Bext',                       # B outside lesion.+ 
'tbp_lv_C',                          # Chroma inside  lesion.+
'tbp_lv_Cext',                       # Chroma outside lesion.+
'tbp_lv_H',                          # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
'tbp_lv_Hext',                       # Hue outside lesion.+
'tbp_lv_L',                          # L inside lesion.+
'tbp_lv_Lext',                       # L outside lesion.+
'tbp_lv_areaMM2',                    # Area of lesion (mm^2).+
'tbp_lv_area_perim_ratio',           # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
'tbp_lv_color_std_mean',             # Color irregularity, calculated as the variance of colors within the lesion's boundary.
'tbp_lv_deltaA',                     # Average A contrast (inside vs. outside lesion).+
'tbp_lv_deltaB',                     # Average B contrast (inside vs. outside lesion).+
'tbp_lv_deltaL',                     # Average L contrast (inside vs. outside lesion).+
'tbp_lv_deltaLB',                    #
'tbp_lv_deltaLBnorm',                # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
'tbp_lv_eccentricity',               # Eccentricity.+
'tbp_lv_minorAxisMM',                # Smallest lesion diameter (mm).+
'tbp_lv_nevi_confidence',            # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
'tbp_lv_norm_border',                # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
'tbp_lv_norm_color',                 # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
'tbp_lv_perimeterMM',                # Perimeter of lesion (mm).+
'tbp_lv_radial_color_std_max',       # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
'tbp_lv_stdL',                       # Standard deviation of L inside  lesion.+
'tbp_lv_stdLExt',                    # Standard deviation of L outside lesion.+
'tbp_lv_symm_2axis',                 # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
'tbp_lv_symm_2axis_angle',           # Lesion border asymmetry angle.+
'tbp_lv_x',                          # X-coordinate of the lesion on 3D TBP.+
'tbp_lv_y',                          # Y-coordinate of the lesion on 3D TBP.+
'tbp_lv_z',                          # Z-coordinate of the lesion on 3D TBP.+
"lesion_size_ratio",             # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
"lesion_shape_index",            # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
"hue_contrast",                  # tbp_lv_H                - tbp_lv_Hext              abs
"luminance_contrast",            # tbp_lv_L                - tbp_lv_Lext              abs
"lesion_color_difference",       # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt  
"border_complexity",             # tbp_lv_norm_border      + tbp_lv_symm_2axis
#"color_uniformity",              # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max

"3d_position_distance",          # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
"perimeter_to_area_ratio",       # tbp_lv_perimeterMM      / tbp_lv_areaMM2
"area_to_perimeter_ratio",       # tbp_lv_areaMM2          / tbp_lv_perimeterMM
"lesion_visibility_score",       # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
# "combined_anatomical_site"      # anatom_site_general     + "_" + tbp_lv_location ! categorical feature
"symmetry_border_consistency",   # tbp_lv_symm_2axis       * tbp_lv_norm_border
"consistency_symmetry_border",   # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)

"color_consistency",             # tbp_lv_stdL             / tbp_lv_Lext
"consistency_color",             # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
"size_age_interaction",          # clin_size_long_diam_mm  * age_approx
"hue_color_std_interaction",     # tbp_lv_H                * tbp_lv_color_std_mean
"lesion_severity_index",         # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
"shape_complexity_index",        # border_complexity       + lesion_shape_index
"color_contrast_index",          # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm

"log_lesion_area",               # tbp_lv_areaMM2          + 1  np.log
"normalized_lesion_size",        # clin_size_long_diam_mm  / age_approx
"mean_hue_difference",           # tbp_lv_H                + tbp_lv_Hext    / 2
"std_dev_contrast",              # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
"color_shape_composite_index",   # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
"3d_lesion_orientation",         # tbp_lv_y                , tbp_lv_x  np.arctan2
"overall_color_difference",      # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3

"symmetry_perimeter_interaction",# tbp_lv_symm_2axis       * tbp_lv_perimeterMM
"comprehensive_lesion_index",    # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
"color_variance_ratio",          # tbp_lv_color_std_mean   / tbp_lv_stdLExt
"border_color_interaction",      # tbp_lv_norm_border      * tbp_lv_norm_color
"size_color_contrast_ratio",     # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
"age_normalized_nevi_confidence",# tbp_lv_nevi_confidence  / age_approx
"color_asymmetry_index",         # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max

"3d_volume_approximation",       # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
"color_range",                   # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
"shape_color_consistency",       # tbp_lv_eccentricity     * tbp_lv_color_std_mean
"border_length_ratio",           # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
"age_size_symmetry_index",       # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
#"index_age_size_symmetry",      # age_approx              * sqrt(tbp_lv_areaMM2 * tbp_lv_symm_2axis)
"index_age_size_symmetry"]       # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis


cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]


final_csv_path = 'C:\\Users\\lsj\\Desktop\\YB\\TBP\\train_data\\filtered_train-metadata.csv' ##### CHANGE THIS PATH TO YOUR PATH! LAST NAME WILL BE train_data\\filtered_train-metadata.csv or /


final_img_path = 'C:\\Users\\lsj\\Desktop\\YB\\TBP\\train_data\\filtered_train-image' ##### CHANGE THIS PATH TO YOUR PATH! LAST NAME WILL BE train_data\\filtered_train-image or /


def get_train_img_path(image_id):
    return f"{TRAIN_DIR}/{image_id}.jpg"


def find_infinite_and_nan(df, columns):
    infinite_values = df[columns].map(np.isinf)
    nan_values = df[columns].isna()
    any_infinite = infinite_values.any()
    any_nan = nan_values.any()
    
    if any_infinite.any() or any_nan.any():
        print("Infinite or NaN values found:")
        for col in columns:
            if any_infinite[col]:
                print(f"Column '{col}' contains infinite values")
                print(df[infinite_values[col]])
            if any_nan[col]:
                print(f"Column '{col}' contains NaN values")
                print(df[nan_values[col]])
    else:
        print("No infinite or NaN values found.")
      

def feature_engineering(df:pd.DataFrame, num_cols, cat_cols):
    df["lesion_size_ratio"]              = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"]             = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"]                   = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"]             = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"]        = np.sqrt(df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"]              = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    #df["color_uniformity"]               = df["tbp_lv_color_std_mean"] / df["tbp_lv_radial_color_std_max"]
    
    df["3d_position_distance"]           = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2) 
    df["perimeter_to_area_ratio"]        = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["area_to_perimeter_ratio"]        = df["tbp_lv_areaMM2"] / df["tbp_lv_perimeterMM"]
    df["lesion_visibility_score"]        = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]
    df["combined_anatomical_site"]       = df["anatom_site_general"] + "_" + df["tbp_lv_location"]
    df["symmetry_border_consistency"]    = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["consistency_symmetry_border"]    = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"] / (df["tbp_lv_symm_2axis"] + df["tbp_lv_norm_border"])
    
    df["color_consistency"]              = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]
    df["consistency_color"]              = df["tbp_lv_stdL"] * df["tbp_lv_Lext"] / (df["tbp_lv_stdL"] + df["tbp_lv_Lext"])
    df["size_age_interaction"]           = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"]      = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"]          = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"]         = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"]           = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df["tbp_lv_deltaLBnorm"]
    
    df["log_lesion_area"]                = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"]         = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"]            = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"]               = np.sqrt((df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"]    = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df["tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"]          = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"]       = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"]     = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df["tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4
    df["color_variance_ratio"]           = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"]       = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"]      = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]
    df["color_asymmetry_index"]          = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]
    
    df["3d_volume_approximation"]        = df["tbp_lv_areaMM2"] * np.sqrt(df["tbp_lv_x"]**2 + df["tbp_lv_y"]**2 + df["tbp_lv_z"]**2)
    df["color_range"]                    = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs() + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs() + (df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    df["shape_color_consistency"]        = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    df["border_length_ratio"]            = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))
    df["age_size_symmetry_index"]        = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    df["index_age_size_symmetry"]        = df["age_approx"] * df["tbp_lv_areaMM2"] * df["tbp_lv_symm_2axis"]
    
    not_exist_in_infer_columns = ["lesion_id", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5", "mel_mitotic_index", "mel_thick_mm", "tbp_lv_dnn_lesion_confidence"]
    df = df.drop(columns=not_exist_in_infer_columns)
    find_infinite_and_nan(df, num_cols)
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    encoded_cats_df = pd.DataFrame()
    for col in cat_cols:
        le = LabelEncoder()
        encoded_cats_df[col] = le.fit_transform(df[col])
        
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_cats_df], axis=1)
    return df
    
    
if __name__ == '__main__':
    ROOT_DIR = CONFIG["root_dir"]
    TRAIN_DIR = CONFIG["train_dir"]
    RANDOM_STATE = 42
    
    train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
    df = pd.read_csv(os.path.join(ROOT_DIR, 'train-metadata.csv'), low_memory=False)

    print("original> data shape:", df.shape, "/ num of positive patients:", df.target.sum(), "/ num of patients", df["patient_id"].unique().shape)

    # NEED TO FILTER DATA HAS NON VALUE (patient_id, age_approx, sex)
    valid_data_condition = (df['patient_id'].notnull()) & (df['age_approx'].notnull()) & (df['sex'].notnull())
    df_positive = df[(df['target'] == 1) & valid_data_condition].reset_index(drop=True) # 381
    df_negative = df[(df['target'] == 0) & valid_data_condition].reset_index(drop=True)

    df_positive_upsampled = resample(df_positive, 
                            replace=True, 
                            n_samples=len(df_positive) * CONFIG["p_upsample_ratio"],
                            random_state=RANDOM_STATE) # 381 * CONFIG["p_upsample_ratio"]
    n_positive = len(df_positive_upsampled)
    n_negative = n_positive * CONFIG["p:n_ratio"]
    df_negative_downsampled = df_negative.sample(n=n_negative, random_state=RANDOM_STATE).reset_index(drop=True) # 381 * CONFIG["p_upsample_ratio"] * CONFIG["p:n_ratio"]

    df_filtered = pd.concat([df_positive_upsampled, df_negative_downsampled]).reset_index(drop=True)    
    df_filtered['file_path'] = df_filtered['isic_id'].apply(get_train_img_path)
    # print(train_images[0]) -> train-image/image\ISIC_0015670.jpg
    # print(df_filtered['file_path'][0]) -> train-image/image/ISIC_2742835.jpg
    train_images = [path.replace('\\', '/') for path in train_images]
    df_filtered = df_filtered[ df_filtered["file_path"].isin(train_images) ].reset_index(drop=True)
    df_filtered = feature_engineering(df_filtered, num_cols, cat_cols)
    
    train_data_dir = os.path.join(os.getcwd(), 'train_data')
    os.makedirs(train_data_dir, exist_ok=True)
    metadata_path = os.path.join(train_data_dir, 'filtered_train-metadata.csv')
    df_filtered.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")
    
    print("filtered> data shape:", df_filtered.shape, "/ num of positive patients:", df_filtered.target.sum(), "/ num of patients", df_filtered["patient_id"].unique().shape)
    
    filtered_image_dir = os.path.join(train_data_dir, 'filtered_train-image')
    os.makedirs(filtered_image_dir, exist_ok=True)
    for file_path in df_filtered['file_path']:
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(filtered_image_dir, file_name))

    print(f"Copied {len(df_filtered)} images to {filtered_image_dir}")