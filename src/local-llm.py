from llama_cpp import Llama


def main():

    import argparse

    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    llm = Llama(
        model_path="models/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        n_ctx=2048,
        chat_format="chat_template.default",
        verbose=False,
    )

    context = [
        {"cve_id": "CVE-2024-1234", "description": "Buffer overflow in OpenSSL ..."},
        {"cve_id": "CVE-2023-9999", "description": "SQL injection in Apache httpd ..."},
    ]

    ctx_text = "\n".join(f"- {c['cve_id']}: {c['description']}" for c in context)

    messages = [
        {"role": "system", "content": "You are a concise, factual security analyst."},
        {
            "role": "user",
            "content": (
                f"Answer the question using ONLY the following context:\n\n{
                    ctx_text}\n\n"
                "Question: Explain what a CVE is in one sentence."
            ),
        },
    ]

    resp = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=128,
    )

    print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
