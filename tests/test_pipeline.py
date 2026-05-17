import logging
import time
from src.scout.pipeline import run_scout
from src.config import DEFAULT_FEEDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    print("Testing parallel scout pipeline...")
    start_time = time.time()
    
    # Run the pipeline and save it to test filtering logic
    # Let's run just one feed for a quicker test or both
    results = run_scout(feed_urls=["https://hnrss.org/best"], top_k=2, save=True)
    
    end_time = time.time()
    
    print(f"\nPipeline finished in {end_time - start_time:.2f} seconds.")
    print(f"Got {len(results)} top articles.")
    for a in results:
        print(f"- {a.total_score}: {a.title}")
