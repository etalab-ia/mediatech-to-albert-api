# TODO : make it a pytest script
base_url = "https://albert.api.dev.etalab.gouv.fr/v1"
api_key = "**"
client = OpenAI(base_url=base_url, api_key=api_key)
headers = {"Authorization": f"Bearer {api_key}"}


prompt = "Je veux des infos sur la médaille d'honneur du travail"
data = {"collections": [**], "k": 6, "prompt": prompt, "method": "semantic"}
response = requests.post(url=f"{base_url}/search", json=data, headers=headers)
response.raise_for_status()
results = response.json()["data"]

print("Chunk returned by the semantic search:")

for result in results:
    print("Content:", result["chunk"]["content"], end="\n\n")
    print("Metadata:", result["chunk"]["metadata"])
    print("\n")



prompt_template = "Answer following question using available documents: {prompt}\n\nDocuments :\n\n{chunks}"

chunks = "\n\n\n".join([result["chunk"]["content"] for result in results])
prompt = prompt_template.format(prompt=prompt, chunks=chunks)

response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model=model,
    stream=False,
    n=1,
)
print(f"RAG response: {response.choices[0].message.content}")