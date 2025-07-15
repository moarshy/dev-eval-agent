# 🔧 DevAgent - AI-Powered Developer Tool Testing Pipeline

**Automatically crawl, analyze, test, and generate intelligent reports for any developer tool documentation.**

DevAgent is a comprehensive testing system that uses AI to evaluate developer tools by analyzing their documentation, generating test cases, executing them, and providing detailed insights for improvement.

## 🌟 What Does DevAgent Do?

DevAgent automates the entire process of evaluating developer tools and APIs:

### 🕷️ **Intelligent Web Crawling**
- Uses **Crawl4AI** with deep crawling strategies to discover documentation pages
- Supports multiple crawling modes: simple, deep (BFS/DFS), and adaptive crawling
- Smart URL normalization to avoid duplicates (`/api` vs `/api/`)
- Filters and focuses on relevant documentation content

### 🧠 **AI-Powered Content Analysis** 
- **DSPy-powered** document analysis that extracts:
  - API operations and capabilities
  - Authentication methods and requirements
  - Usage patterns and workflows
  - Error scenarios and edge cases
  - Code examples and integration guides

### 🎯 **Automated Test Generation**
- Generates comprehensive test cases across multiple categories:
  - **Authentication testing** - API key validation, OAuth flows
  - **Basic usage** - Core functionality verification  
  - **Core workflows** - Multi-step process testing
  - **Error handling** - Edge cases and failure scenarios
- Prioritizes tests based on complexity and importance

### ⚡ **Parallel Test Execution**
- Runs tests in parallel with configurable worker pools
- Thread-safe execution with isolated contexts
- Real-time progress tracking and error reporting
- Graceful fallback to sequential execution when needed

### 📊 **Intelligent Reporting**
- **AI-generated insights** analyzing test failures against documentation
- **Page-level reports** with specific recommendations
- **Overall quality scores** and improvement suggestions  
- **Gap analysis** identifying missing examples and unclear documentation
- Web-based dashboard with modern, interactive UI

## 🚀 Key Features

- **🌐 Modern Web Interface** - FastAPI-powered dashboard with real-time updates
- **🔧 Flexible Configuration** - Customize crawling depth, test parameters, and API keys
- **📈 Progress Tracking** - Monitor pipeline execution across all stages
- **💾 Persistent Results** - Save and review past testing runs
- **🎨 Beautiful UI** - Modern, responsive design with dark theme
- **🔄 Real-time Updates** - Auto-refreshing status and progress indicators
- **📋 Comprehensive Logging** - Detailed execution traces and error reporting

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.11+** 
- **uv** (recommended) or pip for package management

### 1. Clone the Repository

```bash
git clone <repository-url>
cd devagent
```

### 2. Install Dependencies

Using **uv** (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### 3. Install Playwright Browsers

**⚠️ IMPORTANT**: After installing the Python packages, you must install the browser binaries:

```bash
uv run playwright install
```

Or if using pip:
```bash
playwright install
```

This downloads the required browser binaries (Chromium, Firefox, WebKit) that Crawl4AI needs for web scraping.

### 4. Configure AI Models

Set up your preferred AI model by setting environment variables:

```bash
# For Gemini (recommended)
export GEMINI_API_KEY="your-openai-api-key"

# OR create a .env file
echo "GEMINI_API_KEY=your-key-here" > .env
```

## 🎮 Usage

### Web Interface (Recommended)

Start the web server:
```bash
uv run devagent-web
```

Then open your browser to: **http://localhost:8005**

The web interface allows you to:
- Configure tool testing parameters
- Set API keys and context variables  
- Monitor real-time progress
- View comprehensive reports
- Access historical test runs

### Command Line Interface

For programmatic usage:
```bash
uv run devagent-cli
```

Or run directly:
```bash
python agents/test.py
```

## 🎯 Example Usage

### Testing an API Documentation Site

1. **Open the web interface** at http://localhost:8005
2. **Enter tool details**:
   - Tool Name: `OpenWeatherMap API`
   - Base URL: `https://openweathermap.org/api`
3. **Add API keys** (KEY:VALUE format):
   - `OPENWEATHER_API_KEY`: `your-api-key-here`
4. **Configure options**:
   - Max Pages: `20`
   - Max Depth: `3` 
   - Keywords: `api, documentation, guide`
5. **Click "Start Testing Pipeline"**
6. **Monitor progress** in real-time
7. **Review results** including:
   - Overall quality score
   - AI-generated insights
   - Page-level analysis
   - Specific improvement recommendations

## 📊 Pipeline Stages

The testing pipeline consists of 5 main stages:

1. **🕷️ Fetching** - Crawl and discover documentation pages
2. **🔍 Analysis** - AI-powered content extraction and categorization  
3. **📝 Test Planning** - Generate comprehensive test scenarios
4. **⚡ Execution** - Run tests in parallel with isolated contexts
5. **📊 Reporting** - Generate insights and recommendations

Each stage provides detailed progress updates and error handling.

## 🔧 Configuration Options

### Crawling Configuration
- **Max Pages**: Maximum number of pages to crawl (1-100)
- **Max Depth**: How deep to crawl from the base URL (1-5)
- **Keywords**: Focus keywords for relevance scoring
- **URLs to Exclude**: Skip specific URLs or patterns

### Execution Configuration  
- **Max Workers**: Number of parallel workers (1-16)
- **API Keys**: Set testing credentials and context variables
- **Timeouts**: Configure request and execution timeouts

### AI Configuration
- **Model Selection**: Choose between OpenAI GPT or Claude models
- **Analysis Depth**: Configure how thorough the AI analysis should be

## 🎨 Web Interface Features

### 📋 Configuration Form
- **Modern, responsive design** with dark theme
- **Dynamic API key management** - add/remove key-value pairs
- **Advanced options** with collapsible sections
- **Form validation** and user-friendly error messages

### 📊 Results Dashboard
- **Real-time progress tracking** with auto-refresh
- **Interactive report viewing** with expandable sections
- **Search and filtering** for large result sets
- **Export capabilities** for reports and raw data

### 🔄 Pipeline Monitoring
- **Live status updates** during execution
- **Detailed error reporting** with stack traces
- **Stage-by-stage progress** with timing information
- **Background execution** without blocking the UI

### Development Setup

1. **Clone and install** as described above
2. **Install development dependencies**:
   ```bash
   uv sync --dev
   ```
3. **Run tests**:
   ```bash
   uv run pytest
   ```
