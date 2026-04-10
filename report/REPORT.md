# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Kiên Trường (2A202600496)
**Nhóm:** Nhóm 31
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có hướng gần nhau trong embedding space, nghĩa là hai câu có ngữ nghĩa tương đồng cao.

**Ví dụ HIGH similarity:**
- Sentence A: "Machine learning is a subset of artificial intelligence"
- Sentence B: "ML is part of AI research"
- Tại sao tương đồng: Cả hai đều nói về mối quan hệ giữa machine learning và AI

**Ví dụ LOW similarity:**
- Sentence A: "The stock market crashed today"
- Sentence B: "I love cooking pasta for dinner"
- Tại sao khác: Hoàn toàn không cùng chủ đề

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine đo góc (hướng) chứ không quan tâm độ lớn. Với text, tần suất từ khác nhau nhưng hướng ngữ nghĩa quan trọng hơn — hai câu dài ngắn khác nhau vẫn có thể cùng nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Step = chunk_size - overlap = 500 - 50 = 450
> Chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500/450) + 1 = 22 + 1 = **23 chunks**
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Step = 500 - 100 = 400 → chunks = ceil(9500/400) + 1 = **24 chunks**
> Overlap nhiều hơn giúp preserve context tốt hơn giữa các chunks liền kề, đặc biệt quan trọng khi ranh giới chunk có thể cắt giữa câu quan trọng.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** VinBus

**Tại sao nhóm chọn domain này?**
> VinBus là một lĩnh vực mới mẻ và đầy tiềm năng tại Việt Nam — transportation đang được digitalize mạnh. Dataset nhỏ gọn (5,051 ký tự) nhưng chứa nhiều thông tin có cấu trúc (metadata phong phú: company name, founded year, cities, characteristics). Topic này vừa đủ phức tạo để test retrieval quality mà không quá khó để clean và chunk.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | vinbus.md | Wikipedia / web research | 5,051 | source, extension, chunk_index |

**Ghi chú:** Nhóm chỉ sử dụng 1 tài liệu chính (vinbus.md) vì domain VinBus còn mới, nguồn public data còn hạn chế. Tài liệu đã được gán metadata phục vụ retrieval và filtering.

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | "data/vinbus.md" | Xác định document gốc khi cần traceback kết quả |
| extension | string | ".md" | Phân biệt loại file để filter hoặc preprocess khác nhau |
| chunk_index | int | 0, 1, 2... | Giữ thứ tự chunk để reconstruct context liên tục |
| department | string | "engineering" | Filter nhanh các chunks thuộc domain cụ thể (dùng trong search_with_filter) |
| lang | string | "en", "vi" | Hỗ trợ multilingual retrieval hoặc language-specific preprocessing |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên vinbus.md (5,051 ký tự, chunk_size=500):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| vinbus.md | FixedSizeChunker (`fixed_size`) | 12 | 466.8 | ❌ Cắt giữa câu, không respect ngữ pháp |
| vinbus.md | SentenceChunker (`by_sentences`) | 8 | 629.1 | ✅ Giữ nguyên câu, context tốt hơn |
| vinbus.md | RecursiveChunker (`recursive`) | 63 | 78.2 | ⚠️ Quá nhiều chunks nhỏ, context bị phân mảnh |

### Strategy Của Tôi

**Loại:** SentenceChunker

**Mô tả cách hoạt động:**
> Dùng regex pattern `(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n` để detect sentence boundaries (dấu `.`, `!`, `?` theo sau bởi space hoặc newline). Mỗi chunk chứa tối đa `max_sentences_per_chunk` sentences. Edge cases: empty text → return [], text ngắn hơn max_sentences → return as-is.

**Tại sao tôi chọn strategy này cho domain VinBus?**
> VinBus document có cấu trúc rõ ràng theo sections (Organization, Electric Bus System, Deployment History...). SentenceChunker giữ nguyên ngữ pháp và semantic units, rất phù hợp cho factual Q&A. Trong benchmark, SentenceChunker đạt 0.865 avg score — cao hơn RecursiveChunker.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality (avg score) |
|-----------|----------|-------------|------------|--------------------|
| vinbus.md | recursive (baseline) | 63 | 78.2 | 0.858 |
| vinbus.md | **sentence (của tôi)** | 8 | 629.1 | 0.865 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | SentenceChunker | 8.65 | Ngữ pháp được giữ, avg length cao (629) | Chunk count thấp, có thể miss specific facts |
| [Tên thành viên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> FixedSizeChunker là baseline tốt nhất cho VinBus (avg score 0.897). Tuy nhiên SentenceChunker cũng đạt 0.865 — gần với FixedSize. RecursiveChunker không hiệu quả vì document ngắn và đã được chia sections rõ ràng. Với domain factual Q&A như VinBus, FixedSize hoặc Sentence đều phù hợp hơn Recursive.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n` để split sentences. Edge cases: empty text → return [], text ngắn hơn max_sentences → return as-is. Strip whitespace từ mỗi sentence trước khi join.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy với separator priority `["\n\n", "\n", ". ", " ", ""]`. Base case: khi không còn separator hoặc text ≤ chunk_size thì return text. Ngược lại split theo separator hiện tại,递归 gọi cho các part > chunk_size.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: embed mỗi doc content bằng `_embedding_fn`, lưu dict gồm id, doc_id, content, embedding, metadata vào `_store`. `search`: embed query, tính cosine similarity với tất cả stored embeddings, sort descending, return top_k results.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: filter trước bằng `metadata_filter` (giữ chunks thỏa all conditions), sau đó mới tính similarity trên filtered set. `delete_document`: filter `_store` giữ lại những record có `doc_id != doc_id`, return True nếu có record bị xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top_k chunks từ store, join content bằng `\n\n` để tạo context. Prompt structure: "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:" — đơn giản và hiệu quả cho factual Q&A.

### Test Results

```
pytest tests/ -v
============================== 42 passed in 0.80s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | VinBus is an electric public transportation company in Vietnam | VinBus operates fully electric buses designed for city transportation | high | 0.110 | ❌ |
| 2 | VinBus was established in 2019 | The stock market crashed today | low | -0.212 | ✅ |
| 3 | VinBus operates under a non-profit-oriented model | The business focuses on maximizing shareholder profits | high | -0.014 | ❌ |
| 4 | Hanoi is one of the operating cities of VinBus | Ho Chi Minh City has VinBus electric bus deployment | high | -0.114 | ❌ |
| 5 | Electric buses reduce emissions and noise pollution | Diesel buses contribute to urban air pollution | high | -0.106 | ❌ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp 1 và 3 dự đoán "high" nhưng actual scores thấp — điều này bất ngờ vì về mặt ngữ nghĩa chúng liên quan. Nguyên nhân: MockEmbedder dùng MD5 hash để tạo vector ngẫu nhiên, không học semantic similarity thực sự. Điều này cho thấy embeddings chỉ capture similarity khi được trained trên large corpus — mock embeddings không capture meaning, chỉ là deterministic pseudo-random vectors.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | When was VinBus established and what is its parent company? | VinBus was established in 2019. Its parent company is Vingroup, specifically through subsidiary VinBus Ecology Transport Services LLC. |
| 2 | What cities has VinBus deployed or planned electric bus networks in? | VinBus has deployed or planned electric bus networks in three major Vietnamese cities: Hanoi, Ho Chi Minh City, and Phu Quoc. |
| 3 | What are the key characteristics of VinBus electric buses? | VinBus electric buses are battery-powered with zero tailpipe emissions, low noise operation, smart monitoring and safety systems, cashless payment support, and free Wi-Fi and onboard digital services. |
| 4 | How does VinBus's business model differ from typical transport companies? | VinBus operates under a non-profit-oriented model, emphasizing public service over direct profitability, supporting urban infrastructure development and encouraging public transport adoption. This aligns with Vingroup's ecosystem approach to urban development. |
| 5 | What are the main challenges facing VinBus? | The main challenges include high upfront infrastructure cost, charging logistics and battery range limits, dependence on government subsidies common in public transport, and operational profitability constraints. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | When was VinBus established and what is its parent company? | Section 2: Organization — company info | 0.900 | ✅ | "VinBus was established in 2019, and its parent company is Vingroup." |
| 2 | What cities has VinBus deployed or planned electric bus networks in? | Section 5: Operating Cities — list of cities | 0.963 | ✅ | "VinBus has deployed or planned electric bus networks in Hanoi, Ho Chi Minh City, and Phu Quoc." |
| 3 | What are the key characteristics of VinBus electric buses? | Section 3: Key Characteristics — battery-powered, zero emissions | 0.891 | ✅ | Lists 6 key characteristics including battery-powered, zero tailpipe emissions, low noise, smart monitoring, cashless payment, free Wi-Fi |
| 4 | How does VinBus's business model differ from typical transport companies? | Section 9: Business Model — non-profit model | 0.941 | ✅ | "VinBus operates under a non-profit-oriented model, emphasizing public service over direct profitability..." |
| 5 | What are the main challenges facing VinBus? | Section 10: Challenges — infrastructure cost, charging logistics | 0.815 | ✅ | Lists high upfront costs, charging logistics, government subsidies dependence, operational profitability constraints |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Qua việc so sánh chunking strategies với các thành viên, tôi nhận ra FixedSizeChunker tuy đơn giản nhưng lại là baseline tốt nhất cho domain VinBus (0.897 avg score). Điều này dạy tôi rằng không phải lúc nào sophisticated method cũng tốt hơn — đôi khi simple rule-based approach hiệu quả hơn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Các nhóm khác demo cách dùng metadata filter để improve retrieval precision — filter theo `department` hoặc `lang` trước khi similarity search giúp giảm noise rất nhiều. Tôi sẽ áp dụng kỹ thuật này cho các domain có nhiều document types.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tạo thêm 2-3 tài liệu bổ sung (ví dụ: VinBus route map, charging infrastructure) để tăng diversity của corpus. Hiện tại chỉ có 1 document khiến retrieval phụ thuộc quá nhiều vào quality của document đó. Ngoài ra, tôi sẽ thử custom chunker kết hợp section headers làm boundaries.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 8 / 10 |
| Chunking strategy | Nhóm | 12 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 28 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **79 / 100** |
