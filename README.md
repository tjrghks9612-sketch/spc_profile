# FIB-SEM 2-Section Dome Cap Surface Area Tool

PySide6 기반 데스크톱 앱입니다. 가로 단면 FIB-SEM 이미지 1장과 세로 단면 FIB-SEM 이미지 1장에서 사용자가 지정한 ROI 내부의 어두운 mound 구조물을 검출하고, 두 직교 단면의 실제 top boundary profile을 기반으로 3D dome-like surface를 근사 생성한 뒤 cap curved surface area를 계산합니다.

이 도구는 **2-section profile-based 3D approximation tool**입니다. 두 장의 직교 단면만으로 실제 3D 형상을 완전 복원하지 않으며, 실제 3D tomography 결과로 해석하면 안 됩니다.

## 설치 및 실행

```bash
pip install -r requirements.txt
python main.py
```

## 입력 이미지 조건

- FIB-SEM grayscale 이미지가 기본 입력입니다.
- RGB/BGRA 이미지가 들어오면 내부에서 grayscale로 변환합니다.
- ROI 안에는 주변보다 어두운 hill 또는 mound 형태 구조물이 하나 있다고 가정합니다.
- 가로 단면과 세로 단면은 같은 구조물의 직교 단면이어야 합니다.

## Pixel Size 입력

`pixel_size_um`에는 이미지 한 pixel이 실제 몇 um에 해당하는지 입력합니다. 예를 들어 1 pixel = 0.01 um이면 `0.01`을 입력합니다. 모든 CD, Height, surface grid, area 계산은 이 값을 곱한 실제 um 좌표계에서 수행됩니다.

## ROI 지정

1. 가로 단면 이미지를 불러옵니다.
2. 세로 단면 이미지를 불러옵니다.
3. 각 이미지 뷰어에서 마우스로 사각형 ROI를 드래그합니다.
4. 다시 드래그하면 ROI가 갱신됩니다.
5. 분석은 ROI 내부에서만 수행됩니다.

ROI가 없으면 분석은 실행되지 않습니다.

## Threshold Sensitivity

`threshold_sensitivity`는 ROI 내부 grayscale intensity 기준 dark object threshold를 보정합니다.

- 낮은 값: 더 어두운 pixel만 구조물로 포함합니다.
- 높은 값: 상대적으로 덜 어두운 pixel도 구조물로 포함합니다.

검출 contour가 구조물보다 작으면 값을 올리고, 주변 noise까지 포함하면 값을 낮추는 방식으로 조절합니다.

## Z축과 Cap Depth

3D surface의 `Z`는 꼭대기(apex)를 0으로 둔 **depth 좌표**입니다. 아래로 내려갈수록 값이 커집니다.

```text
z_depth_um = section_height_um - height_from_baseline_um
apex z = 0
downward direction = positive
```

`cap_depth_um`은 apex에서 아래로 내려온 깊이입니다.

```text
z_cut_um = min(cap_depth_um, surface_max_depth_um)
cap_mask = Z <= z_cut_um
```

`cap_depth_um`이 surface 최대 depth보다 크면 전체 valid surface가 cap 영역으로 사용됩니다.

## 계산 방식

각 단면에서 다음 값을 추출합니다.

- CD
- Height
- baseline 기준 top boundary height profile

가로 단면 profile은 x-depth profile, 세로 단면 profile은 y-depth profile로 변환됩니다. 각 profile은 CD center가 0이 되도록 정렬됩니다.

3D surface는 단순한 타원체 dome이 아닙니다. 실제 단면에서 추출한 profile을 normalized radial lookup으로 사용하고, 방향각에 따라 가로/세로 profile을 부드럽게 보간합니다. x축 단면은 가로 profile, y축 단면은 세로 profile에 최대한 일치하도록 실제 profile depth를 blending합니다.

even `grid_resolution`에서도 apex가 grid에서 빠지지 않도록 x=0, y=0 좌표를 mesh에 포함합니다. projected area는 평균 grid pitch가 아니라 실제 cell 폭과 높이로 계산합니다.

surface area는 X, Y, Z가 모두 um 단위인 mesh grid에서 계산됩니다. 각 grid cell을 삼각형 2개로 나누고 cross product로 실제 curved area를 합산합니다.

## 결과 CSV 컬럼

`result_summary.csv`에는 다음 컬럼이 저장됩니다.

- `CD_x_um`: 가로 단면 CD
- `CD_y_um`: 세로 단면 CD
- `H_x_um`: 가로 단면 Height
- `H_y_um`: 세로 단면 Height
- `H_global_um`: `H_x_um`과 `H_y_um`의 평균
- `H_surface_max_um`: 생성된 3D surface의 최대 depth
- `cap_depth_um`: 입력 cap depth
- `z_cut_um`: apex 기준 cap cut depth
- `cap_curved_surface_area_um2`: cap curved surface area
- `cap_projected_area_um2`: cap top-view projected area
- `grid_resolution`: surface grid 해상도
- `pixel_size_um`: 입력 pixel size
- `threshold_sensitivity`: threshold sensitivity
- `smoothing_strength`: smoothing strength
- `morph_strength`: morphology strength
- `sanity_check_x_rmse_um`: y=0 단면 depth와 가로 profile depth 비교 RMSE
- `sanity_check_y_rmse_um`: x=0 단면 depth와 세로 profile depth 비교 RMSE

`horizontal_profile.csv`와 `vertical_profile.csv`의 `z_um`은 apex 기준 depth입니다. 추적용으로 `height_from_baseline_um`도 함께 저장합니다.

## 저장 파일

`결과 저장` 버튼을 누르면 앱 실행 위치 아래에 `output/YYYYMMDD_HHMMSS` 폴더가 생성되고 다음 파일이 저장됩니다.

- `horizontal_detection_overlay.png`
- `vertical_detection_overlay.png`
- `horizontal_profile.csv`
- `vertical_profile.csv`
- `surface_grid.npz`
- `result_summary.csv`
- `profiles.png`
- `3d_surface.png`
- `cap_highlighted_3d_surface.png`
- `cap_top_view.png`

`surface_grid.npz`에는 apex 기준 depth인 `Z`/`Z_depth_um`와 baseline 기준 height인 `Z_height_from_baseline_um`가 함께 저장됩니다.

## Synthetic 검증

실제 이미지 없이도 `Synthetic 샘플 생성` 버튼으로 가로/세로 synthetic FIB-like mound 이미지를 생성할 수 있습니다. 두 이미지의 폭은 다르고 높이는 유사하게 생성되며, 기본 ROI가 자동 지정됩니다. 바로 `분석 실행`을 눌러 전체 pipeline을 확인할 수 있습니다.

## Single Section CD-Depth 모드

상단의 `Single Section CD-Depth` 버튼을 누르면 단일 이미지 분석 화면으로 전환됩니다. 기존 2-section 3D cap area 모드는 그대로 유지됩니다.

단일 모드 workflow는 다음과 같습니다.

1. FIB-SEM 단면 이미지를 한 장 또는 여러 장 동시에 불러옵니다.
2. 첫 번째 미리보기 이미지에서 ROI를 드래그합니다.
3. `pixel_size_um`, `max_depth_um`, `depth_step_um`, threshold/smoothing/morphology 값을 설정합니다.
4. `프로파일 분석 실행`을 누릅니다.
5. 모든 이미지에 동일한 ROI와 옵션을 적용해 ROI 내부 dark mound의 top boundary profile을 추출하고, apex 기준 depth별 CD를 계산합니다.
6. `Batch CD-depth 결과 저장`으로 결과를 저장합니다.

단일 모드의 depth 좌표도 apex 기준입니다.

```text
apex depth = 0
downward direction = positive
```

`max_depth_um`은 꼭대기에서 아래로 어디까지 CD를 기록할지 정하는 값입니다. 입력한 값이 실제 profile height보다 크면 실제 height까지만 계산합니다. `depth_step_um`은 depth sampling 간격입니다.

각 depth에서의 CD는 top boundary profile과 해당 depth 수평선의 좌우 교차 위치를 보간해서 계산합니다. 실제 profile 노이즈 때문에 depth가 좌우 edge 방향으로 흔들리는 경우를 줄이기 위해, apex에서 좌/우 edge로 나가는 방향의 depth를 단조 증가 envelope로 보정한 뒤 CD를 계산합니다.

단일 모드 batch 저장 파일은 `output/YYYYMMDD_HHMMSS` 아래에 생성됩니다.

- `batch_result_summary.csv`
- `batch_profile_cd_depth.png`
- `single_overlays/{image}_overlay.png`
- `single_profiles/{image}_profile.csv`
- `cd_by_depth/{image}_cd_by_depth.csv`
- `single_plots/{image}_profile_cd_depth.png`

## 한계

- 이 앱은 실제 3D tomography 도구가 아닙니다.
- 두 단면 사이의 대각선 방향 형상은 두 profile 기반의 방향 보간 결과입니다.
- ROI 내부에 구조물이 여러 개 있거나, 구조물과 배경 contrast가 낮으면 검출이 실패하거나 contour가 부정확할 수 있습니다.
- baseline은 mask flank와 하단 contour를 기반으로 한 단일 y pixel 값으로 근사합니다.
