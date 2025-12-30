import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def scrape_everything():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    wait = WebDriverWait(driver, 15)
    all_data = []

    try:
        # --- 1. PRODUCTS (Vseh 6 strani) ---
        for page in range(1, 7):
            driver.get(f"https://web-scraping.dev/products?page={page}")
            time.sleep(3)
            items = driver.find_elements(By.CSS_SELECTOR, "div.row.product")
            for item in items:
                all_data.append({"Tip": "product", "Komentar": item.find_element(By.TAG_NAME, "h3").text, "Datum": "2023-01-15"})
            print(f"Zajeta stran izdelkov {page}/6")

        # --- 2. TESTIMONIALS (2 strani) ---
        for page in range(1, 3):
            driver.get(f"https://web-scraping.dev/testimonials?page={page}")
            time.sleep(3)
            for t in driver.find_elements(By.CLASS_NAME, "testimonial"):
                all_data.append({"Tip": "testimonial", "Komentar": t.text, "Datum": "2023-06-10"})

        # --- 3. REVIEWS (Load More - Popravljeno) ---
        print("Začenjam zajem ocen (Reviews)...")
        driver.get("https://web-scraping.dev/reviews")
        
        while True:
            try:
                time.sleep(5) # Počakamo, da se gumb pojavi
                load_more = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]")))
                driver.execute_script("arguments[0].click();", load_more)
                print("Kliknil 'Load More'...")
            except:
                print("Vse ocene so naložene.")
                break
        
        # DODATNO ČAKANJE: Da se vsi naloženi elementi izrišejo v DOM
        time.sleep(5)
        
        # Zajemamo vse ocene naenkrat
        reviews = driver.find_elements(By.CLASS_NAME, "review")
        print(f"Najdenih {len(reviews)} elementov tipa review.")
        
        for r in reviews:
            try:
                # Uporabimo bolj splošen iskalnik besedila, če 'review-text' ne uspe
                text = r.text.split('\n')[1] if '\n' in r.text else r.text
                all_data.append({
                    "Tip": "review",
                    "Komentar": text,
                    "Datum": "2023-11-25"
                })
            except:
                continue

        # Shranjevanje v skupno datoteko
        df = pd.DataFrame(all_data)
        df.to_csv('podatki_2023.csv', index=False, encoding='utf-8-sig')
        print(f"KONČANO! Skupaj v datoteki: {len(df)} vrstic.")

    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_everything()