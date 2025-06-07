{
  "nodes": [
    {
      "parameters": {
        "operation": "search",
        "text": "={{ /*n8n-auto-generated-fromAI-override*/ $fromAI('Search_Text', ``, 'string') }}",
        "returnAll": "={{ /*n8n-auto-generated-fromAI-override*/ $fromAI('Return_All', ``, 'boolean') }}",
        "simple": "={{ /*n8n-auto-generated-fromAI-override*/ $fromAI('Simplify', ``, 'boolean') }}",
        "options": {}
      },
      "type": "n8n-nodes-base.notionTool",
      "typeVersion": 2.2,
      "position": [
        680,
        220
      ],
      "id": "b6ae634c-24a0-484f-b21c-317ae5661af8",
      "name": "Notion Search Page",
      "credentials": {
        "notionApi": {
          "id": "2VJBiOZuyVNZc4gg",
          "name": "Notion account"
        }
      }
    }
  ],
  "connections": {
    "Notion Search Page": {
      "ai_tool": [
        []
      ]
    }
  },
  "pinData": {},
  "meta": {
    "templateCredsSetupCompleted": true,
    "instanceId": "1d2666320bb19d8da8975c9b1013bd7bdad6e0bf91ffe1acdc74e48c3285fb2f"
  }
}