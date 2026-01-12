import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'MLOps Training',
  description: 'Hands-on training for MLOps infrastructure, deployment, and CI/CD',

  // Clean URLs (no .html extension)
  cleanUrls: true,

  // Ignore dead links for work-in-progress documentation
  ignoreDeadLinks: true,

  themeConfig: {
    // Site navigation
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Study Guide', link: '/README' },
      { text: 'Module 1', link: '/module-01/README' },
    ],

    // Sidebar configuration
    sidebar: {
      '/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Study Guide', link: '/README' },
          ]
        },
        {
          text: 'Module 1: Infrastructure & Prerequisites',
          collapsed: false,
          items: [
            { text: 'Module Overview', link: '/module-01/README' },
            {
              text: 'AWS Cloud Services',
              collapsed: false,
              items: [
                { text: 'AWS Overview', link: '/module-01/aws/README' },
                { text: 'Cloud Concepts (Domain 1)', link: '/module-01/aws/cloud-concepts' },
                { text: 'Security & Compliance (Domain 2)', link: '/module-01/aws/security-compliance' },
                { text: 'Deployment Methods', link: '/module-01/aws/deployment-methods' },
                { text: 'Compute Services', link: '/module-01/aws/compute-services' },
                { text: 'Storage Services', link: '/module-01/aws/storage-services' },
                { text: 'Database Services', link: '/module-01/aws/database-services' },
                { text: 'Networking Services', link: '/module-01/aws/networking-services' },
                { text: 'Analytics Services', link: '/module-01/aws/analytics-services' },
                { text: 'AI/ML Services', link: '/module-01/aws/ai-ml-services' },
                { text: 'Billing & Pricing (Domain 4)', link: '/module-01/aws/billing-pricing' },
                {
                  text: 'LocalStack Labs',
                  collapsed: true,
                  items: [
                    { text: 'Quick Start', link: '/module-01/aws/localstack/quick-start' },
                    { text: 'Full Guide', link: '/module-01/aws/localstack/guide' },
                    { text: 'Compute Practice', link: '/module-01/aws/localstack/compute' },
                    { text: 'Storage & Database Practice', link: '/module-01/aws/localstack/storage-database' },
                    { text: 'Networking & Analytics Practice', link: '/module-01/aws/localstack/networking-analytics-security' }
                  ]
                }
              ]
            },
            {
              text: 'Terraform',
              collapsed: true,
              items: [
                { text: 'Terraform Basics', link: '/module-01/terraform/basics' },
                { text: 'Terraform Examples', link: '/module-01/terraform/examples' },
                { text: 'Terraform Exercises', link: '/module-01/terraform/exercises' }
              ]
            }
          ]
        }
      ]
    },

    // Social links
    socialLinks: [
      { icon: 'github', link: 'https://github.com/yourusername/mlops-training' }
    ],

    // Footer
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026-present'
    },

    // Edit link
    editLink: {
      pattern: 'https://github.com/yourusername/mlops-training/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    // Last updated text
    lastUpdated: {
      text: 'Last updated',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },

    // Search
    search: {
      provider: 'local'
    }
  },

  // Markdown configurations
  markdown: {
    // Line numbers in code blocks
    lineNumbers: true,

    // Language display
    config: (md) => {
      // Add custom markdown-it plugins if needed
      return md
    }
  },

  // Build optimizations
  vite: {
    build: {
      chunkSizeWarningLimit: 1000
    }
  }
})
