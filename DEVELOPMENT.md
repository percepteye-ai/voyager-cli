# Development Setup

## Quick Start

1. **Install dependencies:**

   ```bash
   npm install
   ```

2. **Build packages:**

   ```bash
   # Build  package
   npm run build && npm i && npm i -g .
   ```

3. **Install globally:**

   ```bash
   npm install -g .
   ```

4. **Run:**
   ```bash
   voy
   ```

## API Gateway Authentication

Set environment variables:

```bash
export API_ENDPOINT="https://your-api-endpoint.com"
export API_AUTH_TOKEN="your-auth-token"
export API_MODEL="gpt-4o"  # optional
```

Then select "Use API Gateway" in the auth dialog.
