"""
LangSmith에서 실행 중인 run을 중단하는 스크립트

사용법:
    python scripts/cancel_langsmith_run.py <run_id>
    python scripts/cancel_langsmith_run.py <run_id> --action rollback
    python scripts/cancel_langsmith_run.py --list  # 최근 실행 목록 보기
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


def list_recent_runs(limit=10):
    """최근 실행 목록을 보여줍니다"""
    try:
        from langsmith import Client
        
        client = Client()
        project_name = os.getenv("LANGCHAIN_PROJECT", "Naver-Stock-Agent")
        
        print("=" * 80)
        print(f"최근 실행 목록 (프로젝트: {project_name})")
        print("=" * 80)
        print()
        
        # 최근 24시간 내의 run 조회
        runs = client.list_runs(
            project_name=project_name,
            limit=limit,
        )
        
        runs_list = list(runs)
        
        if not runs_list:
            print("실행 기록이 없습니다.")
            return
        
        for i, run in enumerate(runs_list, 1):
            status_icon = {
                "success": "✅",
                "error": "❌",
                "pending": "⏳",
                "running": "🔄",
            }.get(run.status, "❓")
            
            print(f"{i}. {status_icon} {run.status.upper()}")
            print(f"   Run ID: {run.id}")
            print(f"   시작: {run.start_time}")
            if run.end_time:
                duration = (run.end_time - run.start_time).total_seconds()
                print(f"   종료: {run.end_time} (소요: {duration:.1f}초)")
            else:
                print(f"   종료: 실행 중...")
            
            if hasattr(run, 'name') and run.name:
                print(f"   이름: {run.name}")
            
            if run.error:
                print(f"   오류: {run.error[:100]}...")
            
            print()
        
    except ImportError:
        print("❌ langsmith 패키지가 설치되지 않았습니다.")
        print("   설치: pip install langsmith")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cancel_run(run_id, action="interrupt"):
    """Run을 중단합니다"""
    try:
        from langsmith import Client
        
        client = Client()
        
        print("=" * 80)
        print(f"Run 중단 중... (action: {action})")
        print("=" * 80)
        print(f"Run ID: {run_id}")
        print()
        
        # Run 정보 확인
        try:
            run = client.read_run(run_id)
            print(f"상태: {run.status}")
            print(f"시작 시간: {run.start_time}")
            
            if run.status in ["success", "error"]:
                print(f"\n⚠️ 이 run은 이미 종료되었습니다 (상태: {run.status})")
                print("중단할 필요가 없습니다.")
                return
            
        except Exception as e:
            print(f"⚠️ Run 정보 확인 실패: {e}")
            print("계속 진행합니다...\n")
        
        # 중단 시도
        if action == "rollback":
            print("⚠️ rollback은 run을 완전히 삭제합니다.")
            confirm = input("계속하시겠습니까? (y/N): ")
            if confirm.lower() != 'y':
                print("취소되었습니다.")
                return
        
        # LangSmith API v2에서는 cancel 방법이 다를 수 있음
        try:
            # 방법 1: update로 상태 변경 시도
            client.update_run(run_id, end_time=datetime.now(), status="error")
            print(f"✅ Run이 중단되었습니다 (ID: {run_id})")
        except AttributeError:
            print("⚠️ 직접 중단 기능을 사용할 수 없습니다.")
            print("LangSmith 대시보드에서 수동으로 중단하세요.")
        
    except ImportError:
        print("❌ langsmith 패키지가 설치되지 않았습니다.")
        print("   설치: pip install langsmith")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='LangSmith run 관리 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/cancel_langsmith_run.py --list
  python scripts/cancel_langsmith_run.py abc123-run-id
  python scripts/cancel_langsmith_run.py abc123-run-id --action rollback
        """
    )
    
    parser.add_argument(
        'run_id',
        nargs='?',
        help='중단할 run의 ID'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        choices=['interrupt', 'rollback'],
        default='interrupt',
        help='중단 방식 (interrupt: 중단, rollback: 삭제)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='최근 실행 목록 보기'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='목록에 표시할 run 개수 (기본값: 10)'
    )
    
    args = parser.parse_args()
    
    # API 키 확인
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ LANGCHAIN_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 확인하세요.")
        sys.exit(1)
    
    try:
        if args.list:
            list_recent_runs(args.limit)
        elif args.run_id:
            cancel_run(args.run_id, args.action)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n작업이 취소되었습니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()

