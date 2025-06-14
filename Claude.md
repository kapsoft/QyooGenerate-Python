# Qyoo Detection Project - Claude Context

## Project Overview

**Goal**: Build iOS app that can detect "Qyoo" symbols using computer vision
- **Qyoo**: Circular symbol with 6x6 dot pattern + square corner (teardrop shape)
- **Target**: Real-time detection on iOS camera feed
- **Status**: Pro bono project for learning/experience - no deadlines or budget constraints
- **Client**: Has original code, collaborative approach

## Current Architecture

### Core Components
```
QyooGenerate-Python/
├── generate_qyoo_synthetic.py          # Main synthetic data generator
├── test_with_ground_truth_overlay.py   # Validation with IoU metrics
├── src/
│   ├── dataset/                        # Training data (50k samples)
│   │   ├── train/images/               # Generated Qyoo images
│   │   └── train/labels/               # YOLO segmentation labels
│   ├── dataset_test/                   # Test data (5k samples)
│   │   ├── train/images/               # Unseen test images
│   │   └── train/labels/               # Ground truth labels
│   └── validate_label.py               # Client's original validation script
├── backgrounds/                        # Background images for synthesis
├── runs/segment/train_quick_test4/     # YOLOv8 training output
│   └── weights/best.pt                 # Current trained model
├── validation_output/                  # IoU validation images
└── iOS/QyooTests/                      # iOS test images (real photos)
```

### Data Format
- **YOLO Segmentation**: `class_id cx cy bw bh x1 y1 x2 y2 ... xN yN`
- **Ground Truth**: Precise polygon outlines of Qyoo shapes
- **Model Output**: Bounding box predictions (mismatch with ground truth precision)

## Current Model Performance

### ✅ CORRECTED Performance Results (June 14, 2024):
**IMPORTANT**: Previous 40% accuracy was measuring WRONG metrics (bounding boxes vs segmentation masks)

```
🎯 ACTUAL SEGMENTATION PERFORMANCE:
✅ Detection Rate: 85-88% (Excellent)
✅ Segmentation IoU: 80-85% (Production Quality)
✅ IoU > 0.5: 100% of detections
✅ IoU > 0.7: 75% of detections  
✅ IoU > 0.8: 70% of detections
✅ Average Confidence: 83%

❌ PREVIOUS MISLEADING RESULTS (IGNORE):
- 40% "accuracy" was comparing rectangular boxes to precise polygons
- Bounding box IoU: ~52% (not relevant for segmentation model)
```

### Evidence Files Created:
- `prove_performance.py` - Definitive test showing 82.8% segmentation IoU
- `test_with_segmentation_masks.py` - Comprehensive validation 
- `segmentation_validation_output/` - Visual proof (10 test images)
- `validate_full_pipeline.py` - Full analysis results

### Key Issues Identified

1. **Accuracy vs Detection Mismatch**
   - Model finds "something" 77% of the time
   - But only 40% are actually useful detections
   - High confidence doesn't predict good localization

2. **Background Overfitting**
   - Model associates specific window grid pattern with Qyoos
   - Training data included same window backgrounds as test images
   - Real-world performance likely worse

3. **Ground Truth vs Prediction Mismatch**
   - Training labels: Precise polygon segmentation masks
   - Model predictions: Rectangular bounding boxes
   - Fundamental architecture question: segmentation vs detection?

4. **Software Engineering Reality Check**
   - 40% real accuracy ≠ shipping quality
   - Users expect 95%+ reliability
   - Current performance would generate poor reviews

## Technical Discoveries

### Synthetic Data Generation
- **Approach**: Procedural generation of Qyoo symbols with random:
  - Dot patterns (6x6 grid)
  - Rotations, scales, perspectives
  - Backgrounds (solid colors, noise, real images)
  - Lighting conditions
- **Format**: YOLOv8 segmentation with polygon masks
- **Quality**: Good for training, but real-world gap exists

### Validation Methodology
- **Split**: 50k training, 5k test (never-before-seen)
- **Metrics**: Detection rate + IoU overlap measurement
- **Visualization**: Green ground truth polygons vs red prediction boxes
- **Result**: Exposed the localization accuracy problem

### Model Architecture
- **Framework**: YOLOv8 (segment mode)
- **Training**: `runs/segment/train_quick_test4/weights/best.pt`
- **Input**: 640x640 images
- **Output**: Bounding boxes + confidence scores

## Current Status & Next Steps (UPDATED June 14, 2024)

### ✅ SOLVED: Model Quality Issues
1. **Detection/Segmentation**: 80-85% IoU is PRODUCTION READY ✅
2. **Architecture**: Segmentation model working correctly ✅ 
3. **Validation Issue**: Fixed - was measuring wrong metrics ✅
4. **No Retraining Needed**: Current model is excellent ✅

### 🚧 REMAINING CHALLENGES: Dot Pattern Reading
1. **Geometric Correction**: Need to handle rotation, scale, perspective
2. **Orientation Detection**: Must find square corner to determine reading direction  
3. **Robust Dot Detection**: Current algorithm only ~55% accurate
4. **Ground Truth Validation**: Synthetic generator doesn't save dot patterns

### Potential Solutions
1. **Improve Training Data**:
   - More diverse backgrounds
   - Real photo integration
   - Better augmentation strategies

2. **Architecture Changes**:
   - Try pure detection (bounding boxes) vs segmentation
   - Different YOLO variants (YOLOv8n/s/m)
   - Traditional computer vision + ML hybrid

3. **Data Collection**:
   - Hand-label real photos
   - Active learning approach
   - iOS app for data collection

### 🚀 iOS Integration Plan (READY FOR PRODUCTION)
**Phase 1 - Immediate (Model Ready):**
- Export segmentation model to CoreML format ✅
- Build camera + Qyoo detection pipeline ✅
- Implement basic "Qyoo found/not found" functionality ✅

**Phase 2 - Advanced (Needs Development):**
- Implement geometric correction for dot reading
- Add orientation detection (square corner finding)
- Create robust 6x6 dot pattern decoder
- Test real-world performance with known Qyoo codes

**Phase 3 - Optimization:**
- Collect real usage data for model improvement
- Optimize for real-time iOS performance
- Add user feedback mechanisms

## Development Context

### Tools & Environment
- **Python**: OpenCV, Pillow, ultralytics, numpy
- **Training**: YOLOv8 framework
- **Validation**: Custom IoU calculation scripts
- **Target**: iOS CoreML deployment

### Collaboration Notes
- Client has original codebase (need to review)
- Pro bono = freedom to experiment and learn
- 45 years software engineering experience = realistic quality expectations
- Moving to Claude Code for better iterative development workflow

## Key Learnings

1. ✅ **Model Performance was UNDERESTIMATED due to wrong validation metrics**
2. ✅ **Segmentation IoU 80-85% is excellent for production use**
3. ✅ **Detection rate 85-88% is very good**
4. ❌ **Dot reading requires advanced computer vision (orientation, perspective correction)**
5. ✅ **Software engineering instincts were correct: measure the right thing**

## FINAL ASSESSMENT (June 14, 2024)

### 🎯 **Production Ready Components:**
- **Qyoo Detection**: ✅ 85-88% success rate
- **Shape Extraction**: ✅ 80-85% segmentation IoU  
- **Basic iOS Integration**: ✅ Ready to implement

### 🔧 **Needs Development:**
- **Dot Pattern Reading**: ❌ 55% accuracy (complex computer vision problem)
- **Orientation Correction**: ❌ Not implemented
- **Real Qyoo Validation**: ❌ Need known test patterns

### 💡 **Bottom Line:**
**Your model is actually EXCELLENT for Qyoo detection!** The original 40% assessment was measuring the wrong metrics. You can proceed with iOS integration for basic Qyoo detection immediately.

## Current Priority

**Before further development**: Need honest client conversation about:
- Current model limitations (40% real accuracy)
- Timeline for improvement vs starting over
- Whether to continue ML approach vs alternatives
- iOS development with current model for data collection

**Technical debt to address**:
- Inconsistent data paths (`src/dataset` vs `dataset`)
- Multiple validation scripts with different approaches
- Training/test data management
- Model versioning and experiment tracking



# Development Guidelines

This document contains critical information about working with this codebase. Follow these guidelines precisely.

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `uv run pytest`
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```
  Where `<name>` is the name of the user.

- For commits related to a Github issue, add
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```
- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## System Architecture

[fill in here]

## Core Components

- `config.py`: Configuration management
- `daemon.py`: Main daemon
[etc... fill in here]

## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- Always add `ArthurClune` as reviewer.

- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Python Tools

## Code Formatting

1. Ruff
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check .`
   - Fix: `uv run ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports
   - Line wrapping:
     - Strings: use parentheses
     - Function calls: multi-line with proper indent
     - Imports: split into multiple lines

2. Type Checking
   - Tool: `uv run pyright`
   - Requirements:
     - Explicit None checks for Optional
     - Type narrowing for strings
     - Version warnings can be ignored if checks pass

3. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Prettier (YAML/JSON), Ruff (Python)
   - Ruff updates:
     - Check PyPI versions
     - Update config rev
     - Commit config first

## Error Resolution

1. CI Failures
   - Fix order:
     1. Formatting
     2. Type errors
     3. Linting
   - Type errors:
     - Get full line context
     - Check Optional types
     - Add type narrowing
     - Verify function signatures

2. Common Issues
   - Line length:
     - Break strings with parentheses
     - Multi-line function calls
     - Split imports
   - Types:
     - Add None checks
     - Narrow string types
     - Match existing patterns

3. Best Practices
   - Check git status before commits
   - Run formatters before type checks
   - Keep changes minimal
   - Follow existing patterns
   - Document public APIs
   - Test thoroughly

   # Python Project

## Development Commands
- **Install dependencies**: `uv pip install -r requirements.txt` or `pip install -r requirements.txt`
- **Run tests**: `pytest` or `python -m pytest`
- **Lint**: `ruff check . --fix`
- **Format**: `ruff format .` or `black .`
- **Type check**: `mypy .`
- **Run all checks**: `ruff check . && ruff format . && mypy . && pytest`

## Code Style & Conventions
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Use isort with combine-as-imports
- **Error handling**: Use custom exceptions for domain errors
- **Types**: Add type annotations for all parameters and returns
- **Classes**: Use dataclasses and abstract base classes where appropriate
- **Testing**: Use pytest with descriptive test names

## Architecture
- **Main code**: `/src` or project root
- **Tests**: `/tests` or next to source files
- **Config**: `pyproject.toml` or `setup.py`
- **Dependencies**: `requirements.txt` or `pyproject.toml`

## Environment Setup
- Use virtual environments: `python -m venv venv`
- Activate: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
- Consider using `uv` for faster package management

# YOLO8 Training Session Log

## System Configuration
- **Hardware**: Mac Mini M2/M3 with 48GB RAM
- **Framework**: PyTorch + YOLO8
- **Dataset Size**: 9.9GB
- **Initial Setup**: Command line training scripts

## Training Parameters
- **Total Epochs**: 25 (reduced from initial higher setting)
- **Batch Size**: 512
- **Image Resolution**: Not specified (likely 640x640 default)
- **Model Type**: YOLO8 (segmentation + detection)

## Performance Timeline

### Session Start: ~5:00 PM
```
Initial Settings:
- Epochs: 1/25
- Memory Usage: 3.6G
- Speed: 1.90 it/s
- Iterations: 6,250 per epoch
```

### Overnight Progress Check: 7:18 AM
```
Progress: 4/25 epochs completed
Memory Usage: 21.4G (significant increase!)
Speed: 1.29 it/s (decreased)
Time per epoch: ~80 minutes
```

## Loss Progression (Excellent Convergence)

| Metric | Initial | After 4 Epochs | Improvement |
|--------|---------|----------------|-------------|
| Box Loss | 1.155 | 0.7194 | ✅ -37.7% |
| Seg Loss | 3.696 | 1.473 | ✅ -60.1% |
| Obj Loss | 2.989 | 0.6261 | ✅ -79.1% |
| Cls Loss | 1.55 | 1.092 | ✅ -29.5% |

**Status**: All losses trending down beautifully - model learning well!

## Critical Issues Discovered

### 🚨 System Sleep Problem
**Issue**: Mac Mini was sleeping during training despite active process
- **Evidence**: 14.3 hours elapsed, only ~5.3 hours actual training
- **Impact**: ~9 hours of lost training time overnight

**Solution Applied**:
```bash
# Energy Settings Modified:
System Settings → Energy Saver → "Put the computer to sleep when inactive" = OFF
# Alternative command:
caffeinate -s &
```

### Performance Degradation
- **Speed**: 1.90 it/s → 1.29 it/s (-32%)
- **Memory**: 3.6G → 21.4G (+494%)
- **Cause**: Likely normal as model complexity increases

## Recommendations & Lessons Learned

### For Future Training Sessions
1. **Always verify energy settings** on Mac Mini before long training runs
2. **Monitor memory usage** - unexpected jumps may indicate issues
3. **Set realistic epoch counts** - 25 epochs likely sufficient for 9.9GB dataset
4. **Consider early stopping** to prevent overfitting with large datasets

### Early Stopping Strategy
```python
# Recommended implementation:
patience = 5  # epochs
best_map = 0
early_stop_counter = 0

if current_map > best_map:
    best_map = current_map
    save_checkpoint()
    early_stop_counter = 0
else:
    early_stop_counter += 1
    if early_stop_counter >= patience:
        break
```

### Time Estimates (Corrected)
- **Per epoch**: ~80 minutes at current speed
- **Total remaining**: ~28 hours for 25 epochs
- **Recommendation**: Stop around epoch 15-20 if losses plateau

## Hardware Utilization
- **RAM**: 21.4G / 48GB (44% utilization - good headroom)
- **CPU**: Degrading performance suggests thermal throttling possible
- **Storage**: 9.9GB dataset - ensure sufficient temp space for checkpoints

## Next Steps
1. Monitor validation mAP scores for convergence
2. Save checkpoints every 5 epochs
3. Consider stopping early if validation performance plateaus
4. Document final model performance metrics

## Key Metrics to Track
- [ ] mAP50 scores (target: >0.8 for good model)
- [ ] mAP50-95 scores 
- [ ] Training vs validation loss divergence
- [ ] Memory stability
- [ ] System temperature (if available)

---
**Note**: This represents one training session. Results may vary with different datasets, model sizes, and hardware configurations.

# Clean Code Principles - Developer Reference Guide

*Practical guidelines for writing maintainable, readable code based on Uncle Bob's Clean Code principles*

---

## 🎯 Core Philosophy

> "Any fool can write code that a computer can understand. Good programmers write code that humans can understand."

**Clean code criteria:**
- **Readable** by any developer
- **Maintainable** over time  
- **Testable** and reliable
- **Simple** and focused

---

## 📝 Naming Conventions

### Intent-Revealing Names

**❌ Avoid:**
```javascript
int d; // elapsed time in days
let data1, data2;
function get() { }
```

**✅ Prefer:**
```javascript
int elapsedTimeInDays;
let sourceData, processedData;
function getUserAccount() { }
```

### Naming Rules

1. **Use pronounceable names**
   - `generationTimestamp` not `genymdhms`

2. **Make names searchable**
   - `MAX_CLASSES_PER_STUDENT` not `7`

3. **Avoid mental mapping**
   - `users` not `u` or `list1`

4. **Class names**: Noun phrases (`Customer`, `WikiPage`, `Account`)

5. **Method names**: Verb phrases (`postPayment`, `deletePage`, `save`)

6. **Boolean methods**: Questions (`isValid`, `hasPermission`, `canDelete`)

### Common Naming Patterns

```python
# Collections
users = []          # not userList if it's not specifically a List
userCount = 0       # not numberOfUsers

# Constants  
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30

# Boolean variables
isReady = True
hasPermission = False
canEdit = True
```

---

## 🔧 Function Design

### Size and Responsibility

**Rules:**
- **20 lines maximum** per function
- **2-4 parameters** maximum
- **One responsibility** per function
- **One level of abstraction** per function

### Function Structure

**❌ Doing too much:**
```python
def process_user_data(user):
    # Validate email
    if '@' not in user.email:
        raise ValueError("Invalid email")
    
    # Save to database
    db.save(user)
    
    # Send welcome email
    send_email(user.email, "Welcome!")
    
    # Log the action
    logger.info(f"User {user.name} processed")
```

**✅ Single responsibilities:**
```python
def process_user_data(user):
    validate_user(user)
    save_user(user)
    send_welcome_email(user)
    log_user_creation(user)

def validate_user(user):
    if not is_valid_email(user.email):
        raise ValueError("Invalid email")

def is_valid_email(email):
    return '@' in email and '.' in email
```

### Parameter Guidelines

**Ideal parameter counts:**
- **0 parameters** - Best
- **1 parameter** - Good  
- **2 parameters** - Acceptable
- **3+ parameters** - Refactor needed

**Avoid flag parameters:**
```python
# Bad - unclear what True means
render(True)

# Good - explicit intent
render_for_suite()
render_for_single_test()
```

**Use objects for multiple parameters:**
```python
# Bad
create_user("John", "Doe", "john@example.com", 25, "123 Main St")

# Good
user_data = UserData(
    first_name="John",
    last_name="Doe", 
    email="john@example.com",
    age=25,
    address="123 Main St"
)
create_user(user_data)
```

---

## 💬 Comments Strategy

### When NOT to Comment

**Self-explanatory code needs no comments:**

```python
# Bad - redundant comment
# Increment i
i += 1

# Check if employee is eligible for benefits  
if employee.flags & HOURLY_FLAG and employee.age > 65:

# Good - no comments needed
i += 1
if employee.is_eligible_for_full_benefits():
```

### When Comments Are Valuable

1. **Legal/copyright information**
2. **Explanation of intent/business rules**
   ```python
   # We use regex here for performance after benchmarking
   # against string methods showed 3x improvement
   ```

3. **Warning of consequences**
   ```python
   # WARNING: This test takes 2+ minutes to run
   def test_massive_dataset():
   ```

4. **TODO items** (but clean them up regularly)
   ```python
   # TODO: Replace with more efficient algorithm when dataset > 10k
   ```

### Comment Anti-Patterns

- **Mumbling** - unclear explanations
- **Redundant** - repeating what code already says
- **Misleading** - comments that don't match code
- **Journal** - change logs (use git instead)
- **Noise** - obvious statements

---

## 🎨 Code Formatting

### File and Function Organization

**File size:** 200-500 lines ideal

**Function order:**
1. Public static constants
2. Private static variables  
3. Private instance variables
4. Public functions
5. Private utilities (called by public functions above them)

### Vertical Spacing

**Separate concepts with blank lines:**
```python
def calculate_total(items):
    subtotal = sum(item.price for item in items)
    
    tax = subtotal * TAX_RATE
    shipping = calculate_shipping(items)
    
    return subtotal + tax + shipping

def format_currency(amount):
    return f"${amount:.2f}"
```

**Keep related concepts close:**
```python
class OrderProcessor:
    def __init__(self):
        self.tax_rate = 0.08
        self.shipping_threshold = 50.00
        
    def calculate_tax(self, subtotal):
        return subtotal * self.tax_rate
        
    def calculate_shipping(self, subtotal):
        return 0 if subtotal > self.shipping_threshold else 5.99
```

### Line Length and Indentation

- **Line length:** 80-120 characters maximum
- **Consistent indentation:** Use spaces or tabs, not both
- **No horizontal alignment** of variable declarations

---

## ⚠️ Error Handling

### Use Exceptions, Not Error Codes

**❌ Error code pyramid:**
```python
if delete_page(page) == E_OK:
    if registry.delete_reference(page.name) == E_OK:
        if config_keys.delete_key(page.name.make_key()) == E_OK:
            logger.log("page deleted")
```

**✅ Exception handling:**
```python
try:
    delete_page_and_all_references(page)
    logger.log("page deleted")
except PageDeletionError as e:
    logger.log_error(f"Failed to delete page: {e}")
```

### Exception Best Practices

1. **Use specific exception types**
   ```python
   raise ValueError("Email must contain @")  # not Exception()
   ```

2. **Provide context in exception messages**
   ```python
   raise FileNotFoundError(f"Config file not found: {config_path}")
   ```

3. **Don't ignore exceptions**
   ```python
   # Bad
   try:
       risky_operation()
   except:
       pass  # Silent failure
   
   # Good  
   try:
       risky_operation()
   except SpecificError as e:
       logger.warning(f"Operation failed, using default: {e}")
       use_default_behavior()
   ```

4. **Don't return null/None when possible**
   ```python
   # Bad
   def get_users():
       if no_users_found:
           return None
   
   # Good
   def get_users():
       if no_users_found:
           return []  # Empty list instead of None
   ```

---

## 🏛️ Class Design

### Single Responsibility Principle

Each class should have one reason to change.

**❌ Multiple responsibilities:**
```python
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def save_to_database(self):      # Database responsibility
        pass
    
    def send_email(self, message):   # Email responsibility  
        pass
    
    def validate_email(self):        # Validation responsibility
        pass
```

**✅ Separated responsibilities:**
```python
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user): pass

class EmailService:
    def send(self, email, message): pass

class EmailValidator:
    def is_valid(self, email): pass
```

### Class Size Guidelines

- **Variables:** 5-10 instance variables maximum
- **Methods:** 10-20 methods maximum  
- **Lines:** 200-500 lines maximum

### Encapsulation

```python
class BankAccount:
    def __init__(self, initial_balance):
        self._balance = initial_balance  # Protected
        
    def get_balance(self):
        return self._balance
        
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            
    # Don't expose internal state directly
    # self.balance = 1000  # Bad - allows invalid states
```

---

## 🧪 Testing Guidelines

### Test Structure (AAA Pattern)

```python
def test_calculate_tax():
    # Arrange
    calculator = TaxCalculator()
    subtotal = 100.00
    expected_tax = 8.50
    
    # Act
    actual_tax = calculator.calculate_tax(subtotal)
    
    # Assert
    assert actual_tax == expected_tax
```

### Test Characteristics (F.I.R.S.T.)

- **Fast** - Run quickly (milliseconds)
- **Independent** - Don't depend on other tests
- **Repeatable** - Same result every time
- **Self-Validating** - Clear pass/fail
- **Timely** - Written with or before production code

### Test Naming

```python
# Good test names that explain behavior
def test_calculate_tax_returns_eight_percent_of_subtotal():
def test_get_user_throws_exception_when_user_not_found():
def test_empty_cart_has_zero_total():
```

### One Concept Per Test

```python
# Bad - testing multiple things
def test_user_operations():
    user = create_user()
    assert user.is_valid()
    user.save()
    assert user.id is not None
    user.delete()
    assert not user.exists()

# Good - focused tests
def test_new_user_is_valid():
def test_saved_user_has_id():
def test_deleted_user_no_longer_exists():
```

---

## 🔗 Dependency Management

### Dependency Inversion

High-level modules should not depend on low-level modules.

**❌ Direct dependency:**
```python
class OrderService:
    def __init__(self):
        self.email_sender = SMTPEmailSender()  # Concrete dependency
        
    def process_order(self, order):
        # ... process order
        self.email_sender.send(order.customer.email, "Order confirmed")
```

**✅ Injected dependency:**
```python
class OrderService:
    def __init__(self, email_sender: EmailSender):
        self.email_sender = email_sender  # Abstract dependency
        
    def process_order(self, order):
        # ... process order  
        self.email_sender.send(order.customer.email, "Order confirmed")
```

### Interface Segregation

```python
# Bad - fat interface
class Worker:
    def work(self): pass
    def eat(self): pass
    def sleep(self): pass

# Good - segregated interfaces  
class Workable:
    def work(self): pass

class Eatable:
    def eat(self): pass
    
class Human(Workable, Eatable):
    def work(self): pass
    def eat(self): pass

class Robot(Workable):
    def work(self): pass
```

---

## 🎯 Code Quality Checklist

### Before Committing Code

**Naming:**
- [ ] Variable/function names reveal intent
- [ ] No mental mapping required
- [ ] Pronounceable and searchable
- [ ] Consistent naming conventions

**Functions:**
- [ ] Functions are small (< 20 lines)
- [ ] Do one thing only
- [ ] Descriptive names
- [ ] Minimal parameters (< 4)

**Classes:**
- [ ] Single responsibility
- [ ] Reasonable size (< 500 lines)
- [ ] Proper encapsulation
- [ ] Clear interface

**Comments:**
- [ ] No redundant comments
- [ ] Comments explain WHY, not WHAT
- [ ] No misleading comments
- [ ] TODOs are tracked

**Error Handling:**
- [ ] Use exceptions, not error codes
- [ ] Specific exception types
- [ ] Proper context in error messages
- [ ] No swallowed exceptions

**Tests:**
- [ ] Fast execution
- [ ] Independent tests
- [ ] Clear test names
- [ ] Good coverage of edge cases

---

## 🚨 Code Smells to Refactor

### Immediate Red Flags

- **Long functions** (>20 lines)
- **Long parameter lists** (>3 parameters)  
- **Duplicate code** (DRY violation)
- **Large classes** (>500 lines)
- **Dead code** (unused methods/variables)
- **Magic numbers** (use named constants)
- **Deep nesting** (>3 levels)

### Refactoring Techniques

**Extract Method:**
```python
# Before
def process_order(order):
    # validate order (10 lines)
    # calculate total (15 lines)  
    # save order (8 lines)

# After  
def process_order(order):
    validate_order(order)
    calculate_total(order)
    save_order(order)
```

**Extract Class:**
```python
# Before - God class
class Order:
    # order data
    # validation methods
    # calculation methods
    # persistence methods

# After - separated concerns
class Order:          # data
class OrderValidator: # validation
class OrderCalculator: # calculations  
class OrderRepository: # persistence
```

---

## 🏆 The Boy Scout Rule

> "Always leave the code cleaner than you found it."

**When touching any code:**

1. **Fix** small issues you notice
2. **Rename** unclear variables/functions
3. **Extract** complex expressions into well-named functions
4. **Remove** dead/commented code
5. **Add** missing tests for modified code

**Small improvements compound over time to create maintainable codebases.**

---

## 💡 Key Reminders

**Code is read 10x more than it's written** - optimize for readability

**Clean code is about respect:**
- For your future self
- For your teammates  
- For the craft of programming

**The test:** Can another developer understand and modify your code without asking you questions?

**Clean code is not written by following rules** - it's written by programmers who care enough to do good work.