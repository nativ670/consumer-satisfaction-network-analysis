// Network estimated using GLASSO with EBIC model selection (gamma=0.5)
// per Epskamp & Fried (2018). Lambda selected: 0.001
// Edge overlap with CV method: 100%, weight correlation: r=1.0000
const nodesData = [
    {
        "id": "smell/fragrance",
        "pageRank": 0.1158
    },
    {
        "id": "price/value",
        "pageRank": 0.1089
    },
    {
        "id": "texture/consistency",
        "pageRank": 0.0915
    },
    {
        "id": "packaging",
        "pageRank": 0.239
    },
    {
        "id": "ingredients",
        "pageRank": 0.2499
    },
    {
        "id": "effectiveness/results",
        "pageRank": 0.0663
    },
    {
        "id": "service/shipping",
        "pageRank": 0.1285
    }
];
const linksData = [
    {
        "source": "packaging",
        "target": "ingredients",
        "value": 0.0778
    },
    {
        "source": "price/value",
        "target": "packaging",
        "value": 0.0652
    },
    {
        "source": "packaging",
        "target": "service/shipping",
        "value": 0.0550
    },
    {
        "source": "smell/fragrance",
        "target": "ingredients",
        "value": 0.0547
    },
    {
        "source": "ingredients",
        "target": "effectiveness/results",
        "value": 0.0518
    },
    {
        "source": "price/value",
        "target": "service/shipping",
        "value": 0.0375
    },
    {
        "source": "ingredients",
        "target": "service/shipping",
        "value": 0.0312
    },
    {
        "source": "texture/consistency",
        "target": "ingredients",
        "value": 0.0298
    },
    {
        "source": "smell/fragrance",
        "target": "texture/consistency",
        "value": 0.0273
    },
    {
        "source": "smell/fragrance",
        "target": "packaging",
        "value": 0.0234
    },
    {
        "source": "texture/consistency",
        "target": "packaging",
        "value": 0.0217
    }
];
