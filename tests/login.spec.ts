import { test, expect } from '@playwright/test';

test('Utilizatorul se poate loga cu succes', async ({ page }) => {
  // 1. Mergem pe pagina de login
  await page.goto('http://localhost:8080/login');

  // 2. Completăm datele de test predefinite pentru login
  await page.fill('input[type="email"]', 'test@checkwise.ro');
  await page.fill('input[type="password"]', 'ParolaTest123!');

  // 3. Apăsăm pe butonul de Login
  await page.click('button[type="submit"]');

  // 4. Verificăm redirecționarea după succes
  await expect(page).toHaveURL('http://localhost:8080/checker'); 
  
  // Opțional: Verificăm un element vizual de confirmare a stării de logat
  const welcomeMessage = page.locator('#welcome-user');
  if (await welcomeMessage.count() > 0) {
    await expect(welcomeMessage).toBeVisible();
  }
});
