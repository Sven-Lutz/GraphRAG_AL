from crawler.core.engine import Engine

keywords = {
    "positive": [
        "klima","klimaschutz","energie","wärme","sanierung","solar",
        "photovoltaik","wind","förder","richtlinie","satzung",
        "ladesäule","elektromobil","nachhalt","mobil","verkehr","rad","gebäude"
    ],
    "negative": [
        "impressum","datenschutz","cookie","login","karriere","stellen","presse"
    ]
}

if __name__ == "__main__":
    seeds = [
        ("muni_1", "https://www.example.de")
    ]
    engine = Engine(keywords)
    engine.run(seeds)
