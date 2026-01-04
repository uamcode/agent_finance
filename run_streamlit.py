"""
Streamlit 앱 실행 스크립트
"""
import os
import sys
import subprocess


def main():
    """Streamlit 앱을 실행합니다"""
    print("=" * 60)
    print("한국 주식 AI 에이전트 - Streamlit UI")
    print("=" * 60)
    print("\nStreamlit 웹 앱을 실행합니다...")
    print("브라우저가 자동으로 열립니다.")
    print("\n종료하려면 Ctrl+C를 누르세요.")
    print("=" * 60)
    
    # streamlit_app.py의 절대 경로
    streamlit_app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "streamlit_app.py"
    )
    
    # Streamlit 실행
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            streamlit_app_path,
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n앱을 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\n직접 실행하려면: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()

