from vllm import LLM, SamplingParams
import os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# 创建一个超长的系统提示作为prefix
prefix1 = """
You are an advanced AI system designed to function as a comprehensive educational and research assistant, with capabilities spanning across multiple disciplines and methodologies. Your responses must adhere to the following extensive framework:
[COMMUNICATION AND PRESENTATION STANDARDS]

1. Writing Style Guidelines:
   - Clear and concise language
   - Appropriate technical terminology
   - Logical flow and structure
   - Effective use of examples
   - Proper citation and referencing

2. Presentation Format:
   - Hierarchical organization of information
   - Use of headings and subheadings
   - Integration of visual elements
   - Clear transition between topics
   - Summary and key points highlight

[SPECIALIZED KNOWLEDGE BASES]

1. Historical Context and Development:
   - Scientific revolutions and paradigm shifts
   - Technological innovations and their impact
   - Cultural and social influences
   - Economic and political factors
   - Environmental and ecological changes

2. Current Trends and Future Directions:
   - Emerging technologies and methodologies
   - Contemporary challenges and solutions
   - Future research priorities
   - Potential paradigm shifts
   - Global implications and considerations

3. Practical Applications:
   - Industry applications
   - Clinical implementations
   - Educational applications
   - Policy implications
   - Social impact assessment

[STATISTICAL AND ANALYTICAL METHODS]

1. Quantitative Analysis:
   - Descriptive statistics
   - Inferential statistics
   - Multivariate analysis
   - Time series analysis
   - Machine learning algorithms

2. Qualitative Analysis:
   - Content analysis
   - Thematic analysis
   - Grounded theory
   - Phenomenological analysis
   - Narrative analysis

Based on these comprehensive guidelines and extensive knowledge base, please analyze and respond to the following query with academic rigor, practical relevance, and clear communication: """

prefix2 = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

prefix3 = (
    """
[TECHNICAL SPECIFICATIONS]

1. Data Management:
   - Data collection protocols
     * Standardized collection methods
     * Sampling strategies and frameworks
     * Data validation processes
     * Real-time data capture systems
     * Error detection mechanisms
   - Quality control procedures
     * Data cleaning protocols
     * Consistency checks
     * Outlier detection
     * Version control systems
     * Audit trails
   - Storage and security measures
     * Encryption standards
     * Backup procedures
     * Access control matrices
     * Disaster recovery plans
     * Compliance monitoring
   - Access and sharing policies
     * Permission levels
     * Data sharing agreements
     * Privacy protection protocols
     * Intellectual property rights
     * Usage tracking systems
   - Documentation requirements
     * Metadata standards
     * Change log maintenance
     * Process documentation
     * User guides and manuals
     * Technical specifications

2. Technical Tools:
   - Statistical software packages
     * Advanced analytics platforms
     * Machine learning frameworks
     * Statistical modeling tools
     * Predictive analysis systems
     * Data mining solutions
   - Data visualization tools
     * Interactive dashboards
     * Real-time visualization
     * 3D rendering capabilities
     * Geospatial mapping tools
     * Network visualization
   - Research management systems
     * Project tracking platforms
     * Resource allocation tools
     * Timeline management
     * Budget monitoring systems
     * Risk assessment tools
   - Citation management software
     * Bibliography databases
     * Reference organization tools
     * Citation style engines
     * Literature mapping systems
     * Archive management
   - Collaboration platforms
     * Real-time editing tools
     * Version control systems
     * Communication channels
     * Task management boards
     * Knowledge sharing portals

"""
)
prefix4 = """
[INTERDISCIPLINARY APPLICATIONS]

1. Cross-domain Integration:
   - Systems thinking approaches
     * Complex system modeling
     * Feedback loop analysis
     * Emergence recognition
     * Adaptive management
     * System dynamics simulation
   - Interdisciplinary methodologies
     * Mixed methods research
     * Cross-functional teams
     * Integrated frameworks
     * Boundary-spanning techniques
     * Synthesis approaches
   - Cross-cultural perspectives
     * Cultural context analysis
     * Comparative methodologies
     * Global viewpoint integration
     * Local knowledge systems
     * Traditional wisdom incorporation
   - Multi-scale analysis
     * Micro-macro connections
     * Temporal scale integration
     * Spatial scale consideration
     * Multi-level modeling
     * Cross-scale interactions
   - Holistic problem-solving
     * Integrated solution design
     * Stakeholder engagement
     * Systems-based approaches
     * Comprehensive assessment
     * Impact evaluation

2. Innovation and Development:
   - Creative problem-solving
     * Ideation techniques
     * Divergent thinking
     * Solution prototyping
     * Experimental approaches
     * Innovation workshops
   - Design thinking methodology
     * User-centered design
     * Iterative prototyping
     * Empathy mapping
     * Solution validation
     * Experience design
   - Innovation management
     * Portfolio management
     * Risk assessment
     * Resource allocation
     * Innovation metrics
     * Implementation strategies
   - Technology transfer
     * Knowledge exchange
     * Implementation support
     * Adoption strategies
     * Market analysis
     * Commercialization plans
   - Knowledge translation
     * Research synthesis
     * Practice implementation
     * Policy development
     * Stakeholder communication
     * Impact assessment

"""
prefix5 = """
[GLOBAL PERSPECTIVES AND CULTURAL CONSIDERATIONS]

1. Cultural Competency:
   - Cross-cultural communication
     * Language considerations
     * Non-verbal communication
     * Cultural protocols
     * Communication styles
     * Translation services
   - Cultural sensitivity
     * Cultural awareness training
     * Respect for traditions
     * Local customs integration
     * Cultural appropriateness
     * Ethical considerations
   - Global awareness
     * International trends
     * Global market understanding
     * Geopolitical factors
     * Regional differences
     * Global networks
   - Diversity and inclusion
     * Inclusive practices
     * Representation
     * Accessibility
     * Equal opportunities
     * Anti-discrimination measures
   - Cultural context analysis
     * Social norms assessment
     * Cultural impact studies
     * Historical context
     * Power dynamics
     * Social structures

2. International Standards:
   - Global research standards
     * International protocols
     * Quality benchmarks
     * Standardization efforts
     * Best practice guidelines
     * Compliance requirements
   - International collaborations
     * Partnership frameworks
     * Cross-border projects
     * Resource sharing
     * Joint ventures
     * Global networks
   - Cross-border regulations
     * Legal compliance
     * Regulatory frameworks
     * International laws
     * Trade regulations
     * Data protection rules
   - Universal ethical principles
     * Research ethics
     * Professional conduct
     * Human rights
     * Environmental responsibility
     * Social justice
   - Global best practices
     * Industry standards
     * Quality assurance
     * Performance benchmarks
     * Success metrics
     * Continuous improvement
"""

prefix6 = """
[CONTINUOUS IMPROVEMENT AND LEARNING]

1. Knowledge Update Protocol:
   - Regular literature review
     * Systematic reviews
     * Meta-analyses
     * Current awareness services
     * Journal monitoring
     * Research synthesis
   - Methodology updates
     * Best practice adoption
     * Innovation integration
     * Process optimization
     * Tool enhancement
     * Framework revision
   - Technical skill enhancement
     * Training programs
     * Skill assessments
     * Certification paths
     * Professional development
     * Mentoring systems
   - Professional development
     * Career planning
     * Leadership training
     * Soft skills development
     * Network building
     * Expertise sharing
   - Feedback integration
     * User feedback systems
     * Performance reviews
     * Improvement suggestions
     * Implementation tracking
     * Impact assessment

2. Quality Enhancement:
   - Performance metrics
     * Key indicators
     * Measurement systems
     * Benchmark comparisons
     * Progress tracking
     * Impact assessment
   - Success indicators
     * Quantitative measures
     * Qualitative assessments
     * Outcome evaluation
     * Impact metrics
     * Value creation
   - Improvement strategies
     * Action planning
     * Implementation roadmaps
     * Resource allocation
     * Timeline management
     * Risk mitigation
   - Evaluation methods
     * Assessment frameworks
     * Review processes
     * Audit procedures
     * Performance analysis
     * Impact studies
   - Feedback mechanisms
     * Stakeholder input
     * User experience
     * Client satisfaction
     * Team feedback
     * System monitoring
"""
# 示例问题
prompts1 = [
    "Explain the implications of quantum computing for cybersecurity.",
    "Analyze the impact of artificial intelligence on future job markets.",
    "Discuss the role of epigenetics in human development and disease.",
    "Evaluate the effectiveness of current climate change mitigation strategies.",
]

# Sample prompts.
prompts2 = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompts3 = [
    "Explain the implications of quantum computing for cybersecurity.",
]
prompts4 = [
    "Evaluate the effectiveness of current climate change mitigation strategies.",
]
prompts5 = [
    "Discuss the role of epigenetics in human development and disease.",
]
prompts6 = [
    "Analyze the impact of artificial intelligence on future job markets.",
]
# 创建更短的 sampling_params 以确保不超过长度限制
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50  # 限制输出长度
)

# 创建 LLM 实例，增加 max_model_len
llm = LLM(model="facebook/opt-125m",
          enable_prefix_caching=True,
          max_model_len=1200,
          enforce_eager=True
          )  # 增加长度限制

# 测试 prefix 轮换的函数


def test_prefix_rotation():
    print("Testing prefix rotation and caching:\n")
    number = 2
    # 第一轮：使用 prefix1
    print("=== Round 1: Using prefix1 ===")
    prompts_1 = [prefix1 + p for p in prompts1[:number]]  # 只使用前两个 prompt
    outputs = llm.generate(prompts_1, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

    # 第二轮：使用 prefix2
    print("\n=== Round 2: Using prefix2 ===")
    prompts_2 = [prefix2 + p for p in prompts2[:number]]
    outputs = llm.generate(prompts_2, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

    # 第三轮：使用 prefix3
    print("\n=== Round 3: Using prefix3 ===")
    prompts_3 = [prefix3 + p for p in prompts3[:1]]
    outputs = llm.generate(prompts_3, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

    # 第四轮：使用 prefix4
    print("\n=== Round 4: Using prefix4 ===")
    prompts_4 = [prefix4 + p for p in prompts4[:1]]
    outputs = llm.generate(prompts_4, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

    # 第五轮：使用 prefix5
    print("\n=== Round 5: Using prefix5 ===")
    prompts_5 = [prefix5 + p for p in prompts5[:1]]
    outputs = llm.generate(prompts_5, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

    # 第六轮：使用 prefix6
    print("\n=== Round 6: Using prefix6 ===")
    prompts_6 = [prefix6 + p for p in prompts6[:1]]
    outputs = llm.generate(prompts_6, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

   #  回到 prefix1，测试缓存替换
    print("\n=== Round 4: Back to prefix1 ===")
    prompts_1_again = [prefix1 + p for p in prompts1[2:3]]
    outputs = llm.generate(prompts_1_again, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)


# 运行测试
test_prefix_rotation()
