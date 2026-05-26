const nodesData = [
    {
        "id": "smell/fragrance",
        "pageRank": 0.1092
    },
    {
        "id": "price/value",
        "pageRank": 0.1363
    },
    {
        "id": "texture/consistency",
        "pageRank": 0.0969
    },
    {
        "id": "packaging",
        "pageRank": 0.2027
    },
    {
        "id": "ingredients",
        "pageRank": 0.2434
    },
    {
        "id": "effectiveness/results",
        "pageRank": 0.1021
    },
    {
        "id": "service/shipping",
        "pageRank": 0.1095
    }
];
const linksData = [
    {
        "source": "smell/fragrance",
        "target": "texture/consistency",
        "value": 0.0357
    },
    {
        "source": "smell/fragrance",
        "target": "packaging",
        "value": 0.0289
    },
    {
        "source": "smell/fragrance",
        "target": "ingredients",
        "value": 0.0777
    },
    {
        "source": "price/value",
        "target": "packaging",
        "value": 0.0725
    },
    {
        "source": "price/value",
        "target": "ingredients",
        "value": 0.043
    },
    {
        "source": "price/value",
        "target": "effectiveness/results",
        "value": 0.0234
    },
    {
        "source": "price/value",
        "target": "service/shipping",
        "value": 0.0481
    },
    {
        "source": "texture/consistency",
        "target": "packaging",
        "value": 0.0241
    },
    {
        "source": "texture/consistency",
        "target": "ingredients",
        "value": 0.039
    },
    {
        "source": "texture/consistency",
        "target": "effectiveness/results",
        "value": 0.0222
    },
    {
        "source": "packaging",
        "target": "ingredients",
        "value": 0.0841
    },
    {
        "source": "packaging",
        "target": "effectiveness/results",
        "value": 0.0219
    },
    {
        "source": "packaging",
        "target": "service/shipping",
        "value": 0.0574
    },
    {
        "source": "ingredients",
        "target": "effectiveness/results",
        "value": 0.0639
    },
    {
        "source": "ingredients",
        "target": "service/shipping",
        "value": 0.0404
    }
];
