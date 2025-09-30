from llama_cpp import Llama
import pandas as pd


def load_model(path: str, verbose=False):

    llm = Llama(
        model_path=path,
        n_ctx=2048,
        chat_format="chat_template.default",
        verbose=verbose,
    )

    return llm


def chat(llm: Llama, messages: list[dict[str, str]]) -> str:

    resp = llm.create_chat_completion(
        messages=messages,
        temperature=0.2,
        max_tokens=128,
    )

    return resp


def message(query: str, df: pd.DataFrame, max_items: int = 8) -> list[dict[str, str]]:

    # cols = ["cve_id", "severity", "published", "cwe", "description"]
    cols = ["cve_id", "description"]
    ctx_df = df.loc[:, cols].head(max_items)

    ctx_text = "\n".join(
        f"- {row.cve_id}: {row.description}" for row in ctx_df.itertuples(index=False)
    )

    return [
        {
            "role": "system",
            "content": "You are a concise, factual security analyst. Use ONLY the provided context. Cite CVE IDs like [CVE-2024-1234]. If context is insufficient, say so.",
        },
        {
            "role": "user",
            "content": f"Context:\n{ctx_text}\n\nQuestion: {
                query}\n\nOutput: a concise answer to the user query",
        },
    ]


def main():

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    context = [
        {"cve_id": "CVE-2024-1234", "description": "Buffer overflow in OpenSSL ..."},
        {"cve_id": "CVE-2023-9999", "description": "SQL injection in Apache httpd ..."},
    ]

    path = "models/llm/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    llm = load_model(path, False)

    df = pd.DataFrame(context)

    prompt = message(args.query, df)

    resp = chat(llm, prompt)

    print(resp["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
