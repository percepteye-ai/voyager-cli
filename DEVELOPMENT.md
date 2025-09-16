# Development Setup

## Quick Start

1. **Install dependencies:**

   ```bash
   npm install
   ```

2. **Build packages:**

   ```bash
   # Build core package
   cd packages/core && npm run build

   # Build CLI package
   cd ../cli && npm run build
   ```

3. **Install globally:**

   ```bash
   npm install -g .
   ```

4. **Run:**
   ```bash
   voyager
   ```

## API Gateway Authentication

Set environment variables:

```bash
export API_ENDPOINT="https://your-api-endpoint.com"
export API_AUTH_TOKEN="your-auth-token"
export API_MODEL="gpt-4o"  # optional
```

Then select "Use API Gateway" in the auth dialog.
