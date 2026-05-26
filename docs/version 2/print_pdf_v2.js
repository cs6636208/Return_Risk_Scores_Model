const puppeteer = require('puppeteer-core');
const path = require('path');

(async () => {
  try {
    console.log("Launching Edge...");
    const browser = await puppeteer.launch({
      executablePath: 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
      headless: "new"
    });
    console.log("Opening new page...");
    const page = await browser.newPage();
    
    // We run this from the project root, so __dirname will be the root.
    // wait, if we run it from docs/version 2, __dirname is docs/version 2
    // Let's use process.cwd() to be safe and join from there
    const rootDir = process.cwd();
    const filePath = path.join(rootDir, 'docs', 'version 2', 'Feature_Comparison_V2.html');
    const fileUrl = 'file:///' + filePath.replace(/\\/g, '/');
    
    console.log(`Navigating to ${fileUrl}`);
    await page.goto(fileUrl, {waitUntil: 'networkidle0'});
    
    const outputPath = path.join(rootDir, 'docs', 'version 2', 'Feature_Comparison_V2.pdf');
    console.log(`Printing PDF to ${outputPath}`);
    await page.pdf({ 
      path: outputPath, 
      format: 'A4',
      printBackground: true,
      margin: {
        top: '20px',
        bottom: '20px',
        left: '20px',
        right: '20px'
      }
    });
    await browser.close();
    console.log("Done!");
  } catch(e) {
    console.error("Error generating PDF:", e);
    process.exit(1);
  }
})();
