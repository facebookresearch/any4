import torch
from datasets import Dataset, load_dataset

prompt1 = """- Fiction: \"Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ...\"
- News: \"The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ...\"
- Code: `public static void main(String[] args) \nSystem.out.println(``Hello world!'');\n`
- Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
- Facts: \"The capital of Egypt is Cairo. It is the largest city in the region and is home to...\"
"""

prompt2 = """- Fiction: \"In a distant kingdom, a young inventor named Leo built a machine that could speak to animals. One evening, a fox arrived with a message that would change everything...\"
- News: \"After months of negotiations, several countries signed a new climate agreement aimed at reducing emissions. Leaders emphasized the importance of swift action and global cooperation...\"
- Code: `def greet(name):\n    print(f\"Hello, {name}! Welcome to the system.\")\n\ngreet(\"Alice\")`
- Math: (8.4 * 3.1) - (12.6 / 2.1) + 7.5 =
- Facts: \"Mount Everest is the highest mountain above sea level, standing at 8,848 meters. It lies on the border between Nepal and the Tibet Autonomous Region of China...\"
"""

prompt3 = """- Fiction: \"The city of Lysara floated above the clouds, tethered only by ancient magic. Every hundred years, a new guardian was chosen—and this time, it was a boy who had never spoken a word...\"
- News: \"Scientists have discovered a potentially habitable exoplanet orbiting a nearby star. The planet, named Kepler-452c, shows signs of liquid water and a stable atmosphere...\"
- Code: `for (int i = 0; i < 5; i++) {\n    System.out.println(\"Iteration: \" + i);\n}`
- Math: A train leaves Station A traveling at 80 km/h. Two hours later, another train departs from the same station at 100 km/h. How long will it take the second train to catch up?
- Facts: \"Honey never spoils. Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible...\"
"""

prompt4 = """- Fiction: \"Beneath the sands of the great desert, an explorer uncovered a glowing map etched in crystal. Following its path led him to a forgotten civilization...\"
- News: \"In a groundbreaking vote, the World Health Organization approved a new global initiative to combat antibiotic resistance, calling it one of the most urgent threats to modern medicine...\"
- Code: `let fruits = [\"apple\", \"banana\", \"cherry\"];\nfruits.forEach(fruit => console.log(fruit));`
- Math: (9.3 - 4.1) * 2.5 + 7.8 / 1.3 =
- Facts: \"The Amazon rainforest produces around 20% of the world's oxygen. It is often referred to as the planet’s lungs and spans across nine countries in South America...\"
"""

prompt5 = """- Fiction: \"Every Tuesday, the moon whispered secrets to Nora through her radio. One night, it warned her that someone else was listening too...\"
- News: \"In an unusual turn of events, a town in northern Sweden elected a moose as its honorary mayor to promote wildlife awareness and tourism...\"
- Code: `function rollDice() {\n  return Math.floor(Math.random() * 6) + 1;\n}\nconsole.log(\"You rolled a\", rollDice());`
- Math: ((12.5 + 3.3) / 2) ** 2 - sqrt(49) =
- Facts: \"Octopuses have three hearts and blue blood. When they swim, two of their hearts actually stop beating, which is why they prefer crawling to swimming...\"
"""


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif data == "c4":
        dataset = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train",
    )
    elif data == "prompt1":
        dataset = Dataset.from_list([{"text": prompt1}])
    elif data == "prompt2":
        dataset = Dataset.from_list([{"text": prompt2}])
    elif data == "prompt3":
        dataset = Dataset.from_list([{"text": prompt3}])
    elif data == "prompt4":
        dataset = Dataset.from_list([{"text": prompt4}])
    elif data == "prompt5":
        dataset = Dataset.from_list([{"text": prompt5}])
    elif data == "prompts":
        dataset = Dataset.from_list([{"text": prompt1}, {"text": prompt2}, {"text": prompt4}, {"text": prompt5}])
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    if n_split > 0:
        return [
            cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
        ]
    else:
        return [cat_samples]
