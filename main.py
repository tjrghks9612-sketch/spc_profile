def main() -> int:
    try:
        from app import run
    except ImportError as exc:
        print("필수 GUI 의존성을 불러오지 못했습니다.")
        print("먼저 다음 명령으로 의존성을 설치하세요:")
        print("  pip install -r requirements.txt")
        print(f"원인: {exc}")
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
