"""
에이전트 그래프 시각화 스크립트

현재 구성된 멀티 에이전트 시스템의 구조를 시각화합니다.

사용법:
    python scripts/visualize_graph.py
    python scripts/visualize_graph.py --output graph.png
    python scripts/visualize_graph.py --format mermaid
"""

import sys
import os
import argparse

# 프로젝트 루트를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)


def visualize_ascii():
    """ASCII 형식으로 그래프 출력"""
    from src.agent import agent
    
    print("=" * 80)
    print("멀티 에이전트 시스템 구조 (ASCII)")
    print("=" * 80)
    print()
    
    try:
        # LangGraph의 ASCII 출력
        graph_repr = agent.get_graph().draw_ascii()
        print(graph_repr)
    except Exception as e:
        print(f"ASCII 출력 실패: {e}")
        print("\n대신 노드 및 엣지 정보를 출력합니다:\n")
        print_graph_structure()


def print_graph_structure():
    """그래프 구조를 텍스트로 출력"""
    from src.agent import agent
    
    graph_data = agent.get_graph()
    
    print("📊 노드 (Agents):")
    print("-" * 80)
    for i, node in enumerate(graph_data.nodes, 1):
        print(f"  {i}. {node}")
    
    print("\n🔗 엣지 (Connections):")
    print("-" * 80)
    for i, edge in enumerate(graph_data.edges, 1):
        source = edge.source if hasattr(edge, 'source') else edge[0]
        target = edge.target if hasattr(edge, 'target') else edge[1]
        print(f"  {i}. {source} → {target}")


def visualize_mermaid(output_file=None):
    """Mermaid 다이어그램 생성"""
    from src.agent import agent
    
    print("=" * 80)
    print("Mermaid 다이어그램 생성")
    print("=" * 80)
    print()
    
    try:
        mermaid_code = agent.get_graph().draw_mermaid()
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(mermaid_code)
            print(f"✅ Mermaid 다이어그램이 저장되었습니다: {output_file}")
            print(f"\n📝 https://mermaid.live 에서 시각화할 수 있습니다.")
        else:
            print(mermaid_code)
            print("\n" + "=" * 80)
            print("📝 위 코드를 복사하여 https://mermaid.live 에 붙여넣으세요.")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ Mermaid 생성 실패: {e}")
        import traceback
        traceback.print_exc()


def visualize_png(output_file="graph.png"):
    """PNG 이미지로 저장"""
    from src.agent import agent
    
    print("=" * 80)
    print("PNG 이미지 생성")
    print("=" * 80)
    print()
    
    try:
        from PIL import Image
        import io
        
        # PNG 생성
        png_data = agent.get_graph().draw_mermaid_png()
        
        # 파일로 저장
        with open(output_file, 'wb') as f:
            f.write(png_data)
        
        print(f"✅ 그래프 이미지가 저장되었습니다: {output_file}")
        
        # 이미지 정보 출력
        image = Image.open(io.BytesIO(png_data))
        print(f"   크기: {image.size[0]} x {image.size[1]} pixels")
        
    except ImportError:
        print("❌ PIL(Pillow) 라이브러리가 필요합니다.")
        print("   설치: pip install pillow")
    except Exception as e:
        print(f"❌ PNG 생성 실패: {e}")
        print("\n대신 Mermaid 형식으로 출력합니다:")
        visualize_mermaid()


def print_agent_details():
    """에이전트 상세 정보 출력"""
    from src.agent import model_name, RAG_AVAILABLE, db_path
    from src.agent.config import MAX_RETRIES
    
    print("\n" + "=" * 80)
    print("에이전트 시스템 상세 정보")
    print("=" * 80)
    print(f"\n🤖 LLM 모델: {model_name}")
    print(f"📚 RAG 사용 가능: {'✅ 예' if RAG_AVAILABLE else '❌ 아니오'}")
    print(f"💾 데이터베이스: {db_path}")
    print(f"🔄 최대 재시도 횟수: {MAX_RETRIES}")
    
    print("\n📋 에이전트 목록:")
    agents = [
        ("Query_interpreter", "사용자 질문 분석 및 전처리"),
        ("supervisor", "전체 워크플로우 관리 및 에이전트 조정"),
        ("SQL_schema_agent", "데이터베이스 스키마 확인"),
        ("SQL_gen_agent", "SQL 쿼리 생성"),
        ("SQL_check_agent", "SQL 쿼리 검증"),
        ("SQL_execute_agent", "SQL 쿼리 실행"),
        ("Final_answer_agent", "최종 답변 생성"),
    ]
    
    if RAG_AVAILABLE:
        agents.insert(3, ("RAG_agent", "기술 용어 검색 (RAG)"))
    
    for i, (name, desc) in enumerate(agents, 1):
        print(f"  {i}. {name:20s} - {desc}")
    
    print("\n🔧 사용 가능한 도구:")
    tools = [
        "sql_db_list_tables - 테이블 목록 조회",
        "sql_db_schema - 테이블 스키마 조회",
        "sql_db_query - SQL 쿼리 실행",
    ]
    
    if RAG_AVAILABLE:
        tools.append("pdf_search - PDF 문서 검색 (기술 용어)")
    
    for tool in tools:
        print(f"  • {tool}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='멀티 에이전트 시스템 구조 시각화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/visualize_graph.py                    # ASCII 출력
  python scripts/visualize_graph.py --format mermaid   # Mermaid 다이어그램
  python scripts/visualize_graph.py --format png       # PNG 이미지 생성
  python scripts/visualize_graph.py --output graph.png # PNG 파일로 저장
        """
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['ascii', 'mermaid', 'png', 'info'],
        default='ascii',
        help='출력 형식 (기본값: ascii)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='출력 파일 경로 (mermaid, png 형식에서 사용)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.format == 'ascii':
            visualize_ascii()
            print_agent_details()
            
        elif args.format == 'mermaid':
            output_file = args.output or 'graph.mmd'
            visualize_mermaid(output_file)
            
        elif args.format == 'png':
            output_file = args.output or 'graph.png'
            visualize_png(output_file)
            
        elif args.format == 'info':
            print_agent_details()
        
        print("\n✅ 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

