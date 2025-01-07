from datetime import date

def main():
    import argparse
    from util import valid_date

    parser = argparse.ArgumentParser()

    parser.add_argument("ref_date", type=valid_date, help="reference date")
    args = parser.parse_args()
    ref_date: date = args.ref_date
    ref_date = ref_date.replace(day=1)

    run(ref_date=ref_date)


if __name__ == "__main__":
    main()
