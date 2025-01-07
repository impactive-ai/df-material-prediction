import dataclasses
from datetime import date


def _valid_date(s):
    try:
        from datetime import datetime

        return datetime.strptime(s, "%Y-%m-%d")  # 날짜 포맷: YYYY-MM-DD
    except ValueError:
        import argparse

        raise argparse.ArgumentTypeError(
            f"Invalid date: '{s}'. Expected format: YYYY-MM-DD"
        )


def _parse_comma_separated(s):
    if s is None:
        return []

    return s.split(",")


@dataclasses.dataclass(frozen=True)
class PredictionParameter:
    ref_date: date
    input_path: str
    output_path: str
    ext_path_list: list[str]


def run_cli():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dt", "-d", type=_valid_date, help="reference date", required=True
    )
    parser.add_argument("--input", "-i", type=str, help="input path", required=True)
    parser.add_argument(
        "--ext",
        "-x",
        type=_parse_comma_separated,
        help="external data path list (comma separated)",
        required=True,
    )
    parser.add_argument("--output", "-o", type=str, help="output path", required=True)

    args = parser.parse_args()

    ref_date: date = args.dt
    ref_date = ref_date.replace(day=1)

    return PredictionParameter(ref_date, args.input, args.output, args.ext)


if __name__ == "__main__":
    print(run_cli())
