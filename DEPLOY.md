# Deploying the live demo (Azure Container Apps)

```
browser ──HTTPS──▶ Azure Container Apps ingress ──▶ container (FastAPI + uvicorn, 1 worker)
                   (0.5 vCPU / 1 GiB, 0–2 replicas)        └─▶ 61M model on CPU, weights baked into the image
GitHub push to main ──▶ Actions: pytest ▶ docker build ▶ ghcr.io/prateek-tiwari10/hinglish-translit:<sha>
                        ▶ az containerapp update (OIDC login) ▶ smoke test of the live URL
```

The image is public on GHCR, so Azure pulls it without registry credentials. The workflow logs in to Azure with a
federated (OIDC) credential, so no password or secret is stored anywhere.

**Live since 2026-11-26:** https://hinglish-translit.gentleground-e8984580.uaenorth.azurecontainerapps.io (UAE North).

No `az` installed? The official CLI image works the same way; the login is kept in `~/.azure`:

```bash
az() { docker run --rm -i --dns 1.1.1.1 -v "$HOME/.azure:/root/.azure" mcr.microsoft.com/azure-cli:2.67.0 az "$@"; }
az login --use-device-code
```

## One-time setup

Run these yourself, in order. The commands are bash. Steps 1–2 happen on GitHub, and steps 3–6 in a shell after `az login`.

### 1. Push to `main` once

The push runs `.github/workflows/deploy.yml`. Its **deploy** job is skipped because the Azure variables don't exist yet, but **test**
and **build** run and push `ghcr.io/prateek-tiwari10/hinglish-translit:<sha>`. Note that sha: it is the commit id shown in the Actions run.

### 2. Make the image public

On GitHub, go to your profile → **Packages** → `hinglish-translit` → **Package settings** → **Change visibility** → **Public**.
The repo is already public, so this exposes nothing new.

### 3. Resource group, environment and app

```bash
az login
az account set --subscription "<subscription name or id>"
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

RG=hinglish-demo-rg
LOC=uaenorth        # see the region note below
ENV=hinglish-env
APP=hinglish-translit
IMAGE=ghcr.io/prateek-tiwari10/hinglish-translit:<sha from step 1>

az group create -n $RG -l $LOC
az containerapp env create -n $ENV -g $RG -l $LOC          # Consumption plan; creates a Log Analytics workspace for the JSON logs

az containerapp create -n $APP -g $RG --environment $ENV \
  --image $IMAGE --target-port 8000 --ingress external \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 0 --max-replicas 2 \
  --scale-rule-name http-load --scale-rule-type http --scale-rule-http-concurrency 4 \
  --env-vars TORCH_THREADS=1 RATE_LIMIT_PER_MIN=30

az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv   # the live URL
```

**Region:** an *Azure for Students* subscription may deploy only to the regions its policy allows. Check with
`az policy assignment list --query "[].parameters.listOfAllowedLocations.value"`. Here those were uaenorth, indonesiacentral,
koreacentral, malaysiawest and indiasouthcentral. Container Apps is not offered in indiasouthcentral, so UAE North is the
closest to India. Central India was refused with `RequestDisallowedByAzure`.

The HTTP scale rule adds a second replica when one replica has more than 4 requests in flight. With min replicas 0, the app
scales to zero when idle, and the first request after a quiet period waits for a cold start.

### 4. A deploy identity for GitHub Actions (OIDC, no secret)

```bash
APP_ID=$(az ad app create --display-name gh-hinglish-deploy --query appId -o tsv)
az ad sp create --id $APP_ID
az ad app federated-credential create --id $APP_ID --parameters '{
  "name": "github-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:Prateek-Tiwari10/HINDI_SLM:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'
az role assignment create --assignee $APP_ID --role Contributor --scope $(az group show -n $RG --query id -o tsv)

echo "AZURE_CLIENT_ID=$APP_ID"
echo "AZURE_TENANT_ID=$(az account show --query tenantId -o tsv)"
echo "AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)"
```

The role is scoped to the one resource group, so the workflow can't touch anything else in the subscription.

### 5. GitHub repo variables

On GitHub, go to Settings → Secrets and variables → Actions → **Variables** (not Secrets: these are IDs, not passwords) and add:

| variable | value |
| --- | --- |
| `AZURE_CLIENT_ID` | from step 4 |
| `AZURE_TENANT_ID` | from step 4 |
| `AZURE_SUBSCRIPTION_ID` | from step 4 |
| `ACA_RESOURCE_GROUP` | `hinglish-demo-rg` |
| `ACA_APP_NAME` | `hinglish-translit` |

### 6. First real deploy

Go to Actions → **deploy-demo** → **Run workflow** on `main`. The deploy job updates the app to the commit's image, waits until `/health`
reports that commit, and then transliterates one sentence through the live URL. From then on, every push to `main` that touches the
server code redeploys automatically.

### 7. Cost guard

- **Free allowance:** Container Apps (Consumption) gives each subscription 180,000 vCPU-s, 360,000 GiB-s and 2M requests a month for free.
  At 0.5 vCPU / 1 GiB that is about 100 hours of active serving a month.
- **Idle cost:** with min replicas 0, idle time costs nothing.
- **One warm replica all month** (min replicas 1) is roughly $10 at the idle rate.
- **Abuse cap:** max replicas 2 and the per-IP rate limit bound what abuse can cost.
- **Azure for Students** has no card attached: when the credit runs out, the subscription is disabled rather than billed.

A $10/month budget emails the subscription owners at 50% and 100% of actual spend, and when the forecast passes 100%:

```bash
SUB=$(az account show --query id -o tsv)
az rest --method put \
  --url "https://management.azure.com/subscriptions/$SUB/providers/Microsoft.Consumption/budgets/hinglish-demo-budget?api-version=2023-05-01" \
  --body '{"properties": {"category": "Cost", "amount": 10, "timeGrain": "Monthly",
    "timePeriod": {"startDate": "2026-11-01T00:00:00Z", "endDate": "2028-10-31T00:00:00Z"},
    "notifications": {
      "actual_50":    {"enabled": true, "operator": "GreaterThanOrEqualTo", "threshold": 50,  "thresholdType": "Actual",     "contactRoles": ["Owner"]},
      "actual_100":   {"enabled": true, "operator": "GreaterThanOrEqualTo", "threshold": 100, "thresholdType": "Actual",     "contactRoles": ["Owner"]},
      "forecast_100": {"enabled": true, "operator": "GreaterThanOrEqualTo", "threshold": 100, "thresholdType": "Forecasted", "contactRoles": ["Owner"]}}}}'
```

## Day to day

**Interview weeks: keep one replica warm** (no cold start; billed while idle at the idle rate):

```bash
az containerapp update -n hinglish-translit -g hinglish-demo-rg --min-replicas 1
```

**Afterwards: back to scale-to-zero:**

```bash
az containerapp update -n hinglish-translit -g hinglish-demo-rg --min-replicas 0
```

**Logs** (JSON lines, one per request with `latency_ms`):

```bash
az containerapp logs show -n hinglish-translit -g hinglish-demo-rg --follow --format text
```

**Swap the model** (e.g. a retrained checkpoint): put the new `owner/repo@revision` in `serve/MODEL_ID` and push. That one line
is the whole change: the image build bakes in the new weights, and the server reads `MODEL_ID`.

**Load test.** Run it either from Actions → **loadtest-demo** → Run workflow with the live URL, or from any machine with Docker using `hey`.
The per-IP rate limit (30/min) would answer most load-test requests with 429, so raise it only for the test and put it back afterwards.
Max replicas is pinned to 1 so the numbers are per replica:

```bash
az containerapp update -n hinglish-translit -g hinglish-demo-rg --set-env-vars RATE_LIMIT_PER_MIN=100000 --min-replicas 1 --max-replicas 1
URL=https://hinglish-translit.gentleground-e8984580.uaenorth.azurecontainerapps.io/api/transliterate
docker run --rm williamyeh/hey -n 100 -c 1 -m POST -T application/json -d '{"text": "main kal office nahi aa paunga"}' $URL
docker run --rm williamyeh/hey -n 200 -c 4 -m POST -T application/json -d '{"text": "main kal office nahi aa paunga"}' $URL
az containerapp update -n hinglish-translit -g hinglish-demo-rg --set-env-vars RATE_LIMIT_PER_MIN=30 --min-replicas 0 --max-replicas 2
```

The 2026-11-26 results are in the README and `results/d7_hinglish/step6_RESULTS.md`.

**Tear everything down:**

```bash
az group delete -n hinglish-demo-rg --yes
```

## Local check (the gate before any cloud step)

```bash
docker build -f serve/Dockerfile -t hinglish-translit .
docker run --rm -p 8000:8000 --cpus 0.5 --memory 1g hinglish-translit
curl -s localhost:8000/health
curl -s -X POST localhost:8000/api/transliterate -H 'Content-Type: application/json' -d '{"text": "kya haal hai"}'
```
