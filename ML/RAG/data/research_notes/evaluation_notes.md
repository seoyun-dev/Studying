# Evaluation Notes

RAG 시스템 평가는 보통 retrieval quality와 answer quality로 나눠서 본다.
retrieval quality는 관련 문서를 잘 찾았는지, answer quality는 검색 근거를 바탕으로 답을 잘 생성했는지를 확인한다.

실험 단계의 PoC에서는 정답셋이 완벽하지 않기 때문에, 대표 질문 몇 개를 골라 수동 평가하는 경우도 많다.
면접에서 PoC를 설명할 때는 "정식 서비스가 아니라 개념 검증 단계였기 때문에, 소수의 대표 질문으로 검색 경로와 답변 일관성을 점검했다"고 말해도 자연스럽다.
