# df-material-prediction

Deepflow 원자재 가격 예측 모델 프로젝트

## 요구사항

- Python 3.10
- Mlflow

## 프로젝트 초기화 방법

```
# global 인터프리터에 virtualenv, mlflow 설치
pip install virtualenv mlflow

# 이 프로젝트를 위한 가상환경 초기화
virtualenv .venv --prompt df-material-prediction

# 가상환경 활성화
source .venv/bin/activate

# 디펜던시 설치
pip install -r requirements.txt
```

## 코딩 룰

- 모든 스크립트들은 mlflow run 명령으로 실행 가능하여야 합니다.
- 읽어들이는 데이터들은 모두 파라메터에 의해 전달되어야 합니다
- train, test 를 나누는 split 지점은 ref_date를 기준으로 상대적으로 정의되어야 합니다.
- 읽어들이는 데이터들은 각 디렉토리 안에 `source/` 디렉토리를 생성한 후 위치 시킵니다 (.gitignore 설정됨)

### 예측 처리 스크립트의 기본 파라메터 정의

- `--dt`: 기준시점. yyyy-MM-dd 형태. 월간 단위 입력시 1일로 입력
- `--input`: 입력 데이터 경로
- `--ext`: 외부 데이터 경로. 여러개 입력시 콤마로 구분하여 선언
- `--output`: 예측값 출력 경로

파이썬 인터프리터 이용시 실행 방법

```shell
cd food
python main.py -d 2024-12-01 \
  -i price_history.parquet \
  -x kosis_monthly_20241201.parquet,krweather_monthly_20241201.parquet \
  -o prediction.csv 
```

Mlflow run 이용시 실행 방법

```shell
cd food
mlflow run . -P dt=2024-12-01 \
  -P input=price_history.parquet \
  -P ext=kosis_monthly_20241201.parquet,krweather_monthly_20241201.parquet \
  -P output=prediction.csv 
```

## 데이터 규격

가격 데이터 식별자(grain_id) 목록은
[원자재 가격 수집 정보](https://docs.google.com/spreadsheets/d/1qnfH9_XJ2fgVKI8OayiOXnExvzFqzXkkYQGiJGAoFRk/edit?usp=sharing)에서 확인 가능하다

### input - 입력 데이터

v(=value)는 이 프로젝트에서 해당 원물의 가격을 의미함.  
데이터의 수집 주기가 월간이었다면 `v_*` 값 모두 동일한 값으로 출력됨

- grain_id: 예측 구분 식별자
- dt: 시점(yyyy-MM-dd 형태)
- v_open: 주기 시작값 (시가)
- v_close: 주기 종료값 (종가)
- v_sum: 합산값
- v_mean: 평균값
- v_min: 최소값
- v_max: 최대값
- v_std: 표준편차

### ext - 외부 데이터

dt 필드는 공통이며 나머지 컬럼들은 사용하는 데이터셋마다 다름

### output - 예측 결과 데이터

[가이드 문서](https://impactive-ai.notion.site/Deepflow-Forecast-14a5aced1ed38059ba11cb6d0bf53432?pvs=4) 참조

## 코드 기여 방법

- 로컬에서 mlflow run 실행하여 정상적으로 동작하는지 확인 하고 커밋 합니다
- Black 을 이용하여 코드 포매팅 후 커밋 합니다
- 사용하지 않는 import 는 제거 합니다
- 작업 브랜치 생성 후 main 브랜치로의 머지 PR을 날려 코드리뷰 후 머지 합니다
- 디펜던시 추가가 있는 경우 프로젝트 루트의 requirements.txt 에 추가하고 자신이 작업하는 프로젝트 디렉토리의 python_env.yaml 에도 추가 합니다.
