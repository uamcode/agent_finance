"""
로그 파일 분석 스크립트
사용법: python scripts/analyze_logs.py
"""
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import sys


def analyze_logs(log_file):
    """로그 파일을 분석하여 통계 생성"""
    
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                continue
    
    print(f"로그 분석 결과: {log_file.name}")
    print(f"총 로그 수: {len(logs)}")
    print()
    
    # 레벨별 통계
    levels = Counter(log['level'] for log in logs)
    print("레벨별 통계:")
    for level, count in levels.items():
        print(f"  {level}: {count}")
    print()
    
    # 에이전트별 실행 시간
    agent_durations = defaultdict(list)
    for log in logs:
        if 'agent_name' in log and 'duration_ms' in log:
            agent_durations[log['agent_name']].append(log['duration_ms'])
    
    if agent_durations:
        print("에이전트별 평균 실행 시간:")
        for agent, durations in sorted(agent_durations.items()):
            avg = sum(durations) / len(durations)
            print(f"  {agent}: {avg:.0f}ms (실행 {len(durations)}회)")
        print()
    
    # 에러 통계
    errors = [log for log in logs if log['level'] == 'ERROR']
    if errors:
        print(f"에러 발생: {len(errors)}건")
        error_types = Counter(log.get('error_type', 'Unknown') for log in errors)
        for error_type, count in error_types.most_common(5):
            print(f"  {error_type}: {count}")
        print()
    
    # 성공/실패율
    sessions = [log for log in logs if 'session_id' in log]
    if sessions:
        completed = len([s for s in sessions if s.get('success') is True])
        failed = len([s for s in sessions if s.get('success') is False])
        total = completed + failed
        if total > 0:
            print(f"세션 성공률: {completed}/{total} ({completed/total*100:.1f}%)")
            print()
    
    # SQL 쿼리 통계
    sql_logs = [log for log in logs if 'query' in log]
    if sql_logs:
        print(f"SQL 쿼리 실행: {len(sql_logs)}회")
        sql_durations = [log['duration_ms'] for log in sql_logs if 'duration_ms' in log]
        if sql_durations:
            avg_sql_duration = sum(sql_durations) / len(sql_durations)
            print(f"  평균 실행 시간: {avg_sql_duration:.0f}ms")
            print(f"  최소 실행 시간: {min(sql_durations)}ms")
            print(f"  최대 실행 시간: {max(sql_durations)}ms")


if __name__ == "__main__":
    # 로그 디렉토리 경로
    log_dir = Path(__file__).parent.parent / "logs"
    
    if not log_dir.exists():
        print("❌ 로그 디렉토리가 없습니다.")
        print(f"   경로: {log_dir}")
        sys.exit(1)
    
    # 가장 최근 로그 파일 분석
    log_files = sorted(log_dir.glob("agent_*.log"))
    
    if not log_files:
        print("❌ 로그 파일이 없습니다.")
        print(f"   경로: {log_dir}")
        sys.exit(1)
    
    # 여러 파일이 있으면 선택할 수 있도록
    if len(log_files) > 1:
        print("📁 로그 파일 목록:")
        for i, f in enumerate(log_files):
            print(f"  {i + 1}. {f.name}")
        print()
    
    # 가장 최근 파일 분석
    analyze_logs(log_files[-1])

